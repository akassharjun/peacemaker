import copy
import functools
import itertools
import logging
import os
import sys
import threading
import time

import pika
import syft as sy
import torch
from modelstore import ModelStore
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from syft.frameworks.torch.fl import utils
from torch import optim, nn
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from db.database_manager import *
from model import Net

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

args = get_fl_config()
device = torch.device("cuda" if args.use_cuda else "cpu")


def generate_virtual_workers(orgs):
    hook = sy.TorchHook(torch)

    worker_list = []

    # create a virtual worker for each organisation
    logging.info("Setting up virtual workers")
    for org in tqdm(orgs):
        if org.name == "Org E":
            continue

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
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True)

    return federated_train_loader, test_loader


def local_step(data, target, local_model, local_optimizer, local_criterion):
    local_model.train()

    local_model.send(data.location)

    if data.location.id == 'Org A' or data.location.id == 'Org B':
        data = torch.flip(data, [0, 1])

    local_optimizer.zero_grad()

    output = local_model(data)

    loss = local_criterion(output, target)

    loss.backward()

    local_optimizer.step()

    return local_model.get(), loss


def train(epoch, worker_dataloader, models, optimizers, criterions):
    for batch_idx, (data, target) in enumerate(worker_dataloader):

        org = data.location.id

        data, target = data.to(device), target.to(device)

        # get the organisation's model
        model = models[org]
        optimizer = optimizers[org]
        criterion = criterions[org]

        model_update, loss = local_step(data, target, model, optimizer, criterion)

        # get back the updated model
        models[org] = model_update

        if batch_idx % 10 == 0:
            loss = loss.get()

            print('Virtual Worker: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                org, epoch,
                batch_idx * args.batch_size,  # no of images done
                len(worker_dataloader) * args.batch_size,  # total images left
                100. * batch_idx / len(worker_dataloader), loss.item()))

    return models


def calculate_test_based_evaluation(models, evaluation_dataloader):
    logging.info("Measuring Contribution")
    contribution_evaluation = {}

    for org, model in tqdm(models.items()):
        contribution_measure = evaluate_contribution_measure(model, evaluation_dataloader)

        contribution_evaluation[org] = contribution_measure

    return contribution_evaluation


def distribute_global_model(virtual_workers, global_model):
    worker_models = {}
    worker_optimizers = {}
    worker_criterions = {}

    logging.info("Distributing Global Model")
    for virtual_worker in tqdm(virtual_workers):
        model = copy.deepcopy(global_model)
        worker_models[virtual_worker.id] = model
        worker_optimizers[virtual_worker.id] = optim.SGD(model.parameters(), lr=args.learning_rate)
        worker_criterions[virtual_worker.id] = nn.CrossEntropyLoss()

    return worker_models, worker_optimizers, worker_criterions


def distribute_payout(contribution_evaluation, budget):
    logging.info("Distributing Payout")

    payout_result = {}
    aggr_contribution = 0

    for org, contribution in contribution_evaluation.items():
        if contribution >= args.minimum_contribution_value / 100:
            aggr_contribution += contribution

    for org, contribution in tqdm(contribution_evaluation.items()):
        payout = 0

        if contribution >= args.minimum_contribution_value / 100:
            payout = contribution / aggr_contribution * budget

        payout_result[org] = {'payout': payout, 'contribution_measure': contribution}

    return payout_result


def collect_participation_fee(organizations):
    logging.info("Collecting Participation Fee")
    fees = 0
    for _ in tqdm(organizations):
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


def calculate_shapley_value(models, evaluation_dataloader):
    # generate possible permutations
    all_perms = list(itertools.permutations(list(models.keys())))
    marginal_contributions = []
    # history map to avoid retesting the models
    history = {}

    for perm in all_perms:
        perm_values = {}
        local_models = {}

        for client_id in perm:
            model = copy.deepcopy(models[client_id])
            local_models[client_id] = model

            # get the current index eg: (A,B,C) on the 2nd iter, the index is (A,B)
            if len(perm_values.keys()) == 0:
                index = (client_id,)
            else:
                index = tuple(sorted(list(tuple(perm_values.keys()) + (client_id,))))

            if index in history.keys():
                current_value = history[index]
            else:
                current_value = evaluate_contribution_measure(model, evaluation_dataloader)
                history[index] = current_value

            perm_values[client_id] = max(0, current_value - sum(perm_values.values()))

        marginal_contributions.append(perm_values)

    sv = {client_id: 0 for client_id in models.keys()}

    # sum the marginal contributions
    for perm in marginal_contributions:
        for key, value in perm.items():
            sv[key] += value

    # compute the average marginal contribution
    sv = {key: value / len(marginal_contributions) for key, value in sv.items()}

    return sv


def initiate_training():
    logging.info("Retrieving Global Model")
    db_global_model = get_global_model()

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

    # load the experiment dataset to the virtual workers
    worker_dataset_loader, evaluation_dataset = load_dataset(virtual_workers)

    budget = collect_participation_fee(organizations)

    worker_models, worker_optimizers, worker_criterions = distribute_global_model(virtual_workers, global_model)

    communication_time = {}

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        worker_models = train(epoch, worker_dataset_loader, worker_models,
                              worker_optimizers, worker_criterions)
        total_time = time.time() - start_time
        communication_time[epoch] = round(total_time, 2)

    if args.evaluation_metric == EvaluationMetric.SHAPLEY_VALUE:
        contribution_evaluation = calculate_shapley_value(worker_models, evaluation_dataset)
    else:
        contribution_evaluation = calculate_test_based_evaluation(worker_models, evaluation_dataset)

    payout_result = distribute_payout(contribution_evaluation, budget)

    federated_model = fed_avg(worker_models)

    # generate global model report
    model_report = test_global_model(federated_model, evaluation_dataset)

    training_round = update_training_round(training_round_id, budget, payout_result, communication_time)

    store_global_model(federated_model, model_report, training_round)


def ack_message(ch, delivery_tag):
    # Channel is open,this message
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message
        pass


# callback for when the peacemaker queue receives a message
def fl_training_callback(ch, delivery_tag, body):
    thread_id = threading.get_ident()
    logging.info('Training Job', thread_id,
                 delivery_tag, body)

    # entry point to FL system
    logging.info("[x] Starting training job")
    initiate_training()

    cb = functools.partial(ack_message, ch, delivery_tag)
    ch.connection.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    # when a message is inserted into the queue, create a thread to
    # run the necessary function for it.
    th = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=fl_training_callback, args=(ch, delivery_tag, body))
    t.start()
    th.append(t)


if __name__ == '__main__':
    # Pika setup for using rabbitmq for subscribing to a queue for training job requests
    try:
        credentials = pika.PlainCredentials('guest', 'guest')

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

        channel.basic_qos(prefetch_count=1)

        threads = []
        on_message_callback = functools.partial(on_message, args=threads)
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
