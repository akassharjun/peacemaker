import copy
import functools
import logging
import os
import sys
import threading
import time

import pika
import syft as sy
import torch
import torch.nn.functional as F
from modelstore import ModelStore
from pika.exchange_type import ExchangeType
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from syft.frameworks.torch.fl import utils
from torch import optim
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from db.database_manager import *
from model import Net

args = get_fl_config()
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

device = torch.device("cuda" if args.use_cuda else "cpu")


def generate_virtual_workers(orgs):
    hook = sy.TorchHook(torch)

    worker_list = []

    # create a virtual worker for each organisation
    logging.info("Setting up virtual workers")
    for org in tqdm(orgs):
        worker_list.append(sy.VirtualWorker(hook, id=f"{org.name}"))

    return worker_list


def evaluate_contribution_measure(model_update, test_loader):
    model_update.eval()

    y_pred = []
    y_true = []
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # iterate over test data
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model_update(data)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        target = target.data.cpu().numpy()
        y_true.extend(target)  # Save Truth

    if args.evaluation_metric == EvaluationMetric.F1_SCORE:
        f1 = f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=1)
        return f1
    elif args.evaluation_metric == EvaluationMetric.PRECISION:
        precision = precision_score(y_true, y_pred, labels=classes, average='weighted', zero_division=1)
        return precision
    elif args.evaluation_metric == EvaluationMetric.RECALL:
        recall = recall_score(y_true, y_pred, labels=classes, average='weighted', zero_division=1)
        return recall

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def test_global_model(global_model, evaluation_dataloader):
    logging.info("Benchmarking aggregated global model")

    global_model.eval()

    y_pred = []
    y_true = []
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # iterate over test data
    for data, target in evaluation_dataloader:
        data, target = data.to(device), target.to(device)

        output = global_model(data)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        target = target.data.cpu().numpy()
        y_true.extend(target)  # Save Truth

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=1)
    precision = precision_score(y_true, y_pred, labels=classes, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, labels=classes, average='weighted', zero_division=1)

    return {"accuracy": accuracy, "f1_score": f1, "precision": precision, "recall": recall}


def load_dataset(virtual_workers):
    logging.info("Loading Datasets")
    federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])).federate(tuple(virtual_workers)),
        batch_size=args['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args['test_batch_size'], shuffle=True)

    return federated_train_loader, test_loader


def local_step(data, target, local_model, local_optimizer):
    local_model.train()

    local_model.send(data.location)

    if data.location.id == 'Org A':
        data = torch.flip(data, [0, 1])

    local_optimizer.zero_grad()

    output = local_model(data)

    loss = F.nll_loss(output, target)

    loss.backward()

    local_optimizer.step()

    return local_model.get(), loss


def train(epoch, worker_dataloader, models, optimizers):
    for batch_idx, (data, target) in enumerate(worker_dataloader):

        org = data.location.id

        data, target = data.to(device), target.to(device)

        # get the organisation's model
        model = models[org]
        optimizer = optimizers[org]

        model_update, loss = local_step(data, target, model, optimizer)

        # get back the updated model
        models[org] = model_update

        if batch_idx % args.log_interval == 0:
            loss = loss.get()

            print('Virtual Worker: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                org, epoch,
                batch_idx * args.batch_size,  # no of images done
                len(worker_dataloader) * args.batch_size,  # total images left
                100. * batch_idx / len(worker_dataloader), loss.item()))

    return models


def measure_contribution(models, evaluation_dataloader):
    logging.info("Measuring Contribution")
    contribution_evaluation = {}
    # for the i-th epoch, get the contribution measure of each organisation
    # and store it to calculate the payout ratio.
    for org, model in tqdm(models.items()):
        contribution_measure = evaluate_contribution_measure(model, evaluation_dataloader)

        contribution_evaluation[org] = contribution_measure

    return contribution_evaluation


def distribute_global_model(virtual_workers, global_model):
    worker_models = {}
    worker_optimizers = {}

    logging.info("Distributing Global Model")
    for virtual_worker in tqdm(virtual_workers):
        model = copy.deepcopy(global_model)
        worker_models[virtual_worker.id] = model
        worker_optimizers[virtual_worker.id] = optim.SGD(model.parameters(), lr=args.learning_rate)

    return worker_models, worker_optimizers


def distribute_payout(contribution_evaluation, budget):
    logging.info("Distributing Payout")

    payout_result = {}
    aggr_contribution = sum(contribution_evaluation.values())

    for org, contribution in tqdm(contribution_evaluation.items()):
        payout = contribution / aggr_contribution * budget

        payout_result[org] = {'payout': payout, 'contribution_measure': contribution}

    return payout_result


def collect_participation_fee(virtual_workers):
    logging.info("Collecting Participation Fee")
    fees = 0
    for _ in tqdm(virtual_workers):
        fees += args.participation_fee
    return fees


def store_global_model(federated_model, model_report, training_round):
    directory = f"{os.getcwd()}/model-store"
    model_store = ModelStore.from_file_system(root_directory=directory)
    domain = 'peacemaker'
    meta_data = model_store.upload(domain, model=federated_model)

    model_id = meta_data["model"]["model_id"]

    save_global_model(model_id, model_report, training_round)


def load_global_model(model_id):
    directory = f"{os.getcwd()}/model-store"
    model_store = ModelStore.from_file_system(root_directory=directory)
    domain = 'peacemaker'
    return model_store.load(domain, model_id)


def fed_avg(models):
    return utils.federated_avg(models)


def initiate_training():
    logging.info("Retrieving Global Model")
    db_global_model = get_global_model()
    global_model = None

    if db_global_model is not None:
        logging.info("Loaded Global Model")
        global_model = load_global_model(db_global_model.model_id)
    else:
        logging.info("No Previous Global Model Found")
        global_model = Net()

    global_model.to(device)

    logging.info("Retrieving Organizations")
    organizations = get_organisations()

    training_round_id = create_training_round(args, organizations)

    virtual_workers = generate_virtual_workers(organizations)

    worker_dataset_loader, evaluation_dataset = load_dataset(virtual_workers)

    budget = collect_participation_fee(virtual_workers)

    worker_models, worker_optimizers = distribute_global_model(virtual_workers, global_model)

    communication_time = {}

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        worker_models = train(epoch, worker_dataset_loader, worker_models,
                              worker_optimizers)
        total_time = time.time() - start_time
        communication_time[epoch] = round(total_time, 2)

    contribution_evaluation = measure_contribution(worker_models, evaluation_dataset)

    payout_result = distribute_payout(contribution_evaluation, budget)

    federated_model = fed_avg(worker_models)

    model_report = test_global_model(federated_model, evaluation_dataset)

    training_round = update_training_round(training_round_id, budget, payout_result, communication_time)

    store_global_model(federated_model, model_report, training_round)


def peacemaker(ch, method, properties, body):
    pass

def ack_message(ch, delivery_tag):
    """Note that `ch` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass

def do_work(ch, delivery_tag, body):
    thread_id = threading.get_ident()
    logging.info('Thread id: %s Delivery tag: %s Message body: %s', thread_id,
                delivery_tag, body)
    # Sleeping to simulate 10 seconds of work
    logging.info("[x] Starting training job")
    initiate_training()

    cb = functools.partial(ack_message, ch, delivery_tag)
    ch.connection.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    thrds = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(ch, delivery_tag, body))
    t.start()
    thrds.append(t)

if __name__ == '__main__':
    try:
        credentials = pika.PlainCredentials('guest', 'guest')
        # Note: sending a short heartbeat to prove that heartbeats are still
        # sent even though the worker simulates long-running work
        parameters = pika.ConnectionParameters(
            'localhost', credentials=credentials, heartbeat=5)
        connection = pika.BlockingConnection(parameters)

        channel = connection.channel()

        channel.exchange_declare(
            exchange="peacemaker_exchange",
            exchange_type="direct",
            passive=False,
            durable=True,
            auto_delete=False)

        channel.queue_declare(queue='peacemaker')

        channel.queue_bind(
            queue="peacemaker", exchange="peacemaker_exchange", routing_key="peacemaker_key")
        # Note: prefetch is set to 1 here as an example only and to keep the number of threads created
        # to a reasonable amount. In production you will want to test with different prefetch values
        # to find which one provides the best performance and usability for your solution
        channel.basic_qos(prefetch_count=1)

        threads = []
        on_message_callback = functools.partial(on_message, args=(threads))
        channel.basic_consume('peacemaker', on_message_callback)

        logging.info('[*] Waiting for training job requests.')

        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            channel.stop_consuming()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        connection.close()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
