import json

import pika
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pika import DeliveryMode
from pydantic import BaseModel

from db.database_manager import *

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


# publishes the training job to the peacemaker message queue
@app.post("/initiate-training")
def initiate_training():
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters('localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.exchange_declare(exchange="peacemaker_exchange",
                             exchange_type='direct',
                             passive=False,
                             durable=True,
                             auto_delete=False)

    channel.basic_publish(
        'peacemaker_exchange', 'peacemaker_key', 'queue:training_job',
        pika.BasicProperties(content_type='text/plain',
                             delivery_mode=DeliveryMode.Transient))

    connection.close()

    return {"result": "Success"}


@app.get("/training-rounds")
def read_training_rounds():
    training_rounds = get_training_rounds()
    response = []

    for tround in training_rounds:
        total_time = 0

        for comm in tround.communication_time:
            total_time += comm.time_taken

        response.append({
            'id': str(tround.id),
            'budget': tround.fl_config.participation_fee * len(tround.organizations),
            'organization_count': len(tround.organizations),
            'epochs': tround.fl_config.epochs,
            'total_time': round(total_time, 2),
            'created_at': tround.created_at
        })

    return response


@app.get("/global-models")
def read_global_models():
    global_models = get_global_models()
    response = []

    for global_model in global_models:

        for value in global_model.model_report:
            global_model.model_report[value] = round(global_model.model_report[value] * 100, 2)

        response.append({
            'id': str(global_model.model_id),
            'organization_count': len(global_model.organizations),
            'model_report': json.loads(global_model.model_report.to_json()),
            'created_at': global_model.created_at
        })

    return response


@app.get("/training-rounds/{org_id}")
def read_org_training_rounds(org_id: str):
    training_rounds = get_training_rounds()

    filtered_training_rounds = []

    for tround in training_rounds:
        if not (org_id == str(org.id) for org in tround.organizations):
            continue

        payout = 0
        contribution = 0

        for payout_info in tround.payout_report:
            if str(payout_info.organization.id) == org_id:
                payout = round(payout_info.payout, 2)
                contribution = payout_info.contribution_measure
                if tround.fl_config.evaluation_metric != EvaluationMetric.SHAPLEY_VALUE:
                    contribution = round(contribution * 100, 2)

        filtered_training_rounds.append(
            {
                'date': tround.created_at,
                'training_round_id': str(tround.id),
                'participation_fee': tround.fl_config.participation_fee,
                'payout': payout,
                'contribution': contribution,
                'evaluation_metric': tround.fl_config.evaluation_metric
            }
        )

    return filtered_training_rounds


@app.get("/fl-config")
def read_fl_config():
    fl_configs = get_fl_configs()

    response = []

    for fl_config in fl_configs:
        response.append({
            'id': str(fl_config.id),
            'batch_size': fl_config.batch_size,
            'test_batch_size': fl_config.test_batch_size,
            'learning_rate': fl_config.learning_rate,
            'epochs': fl_config.epochs,
            'evaluation_metric': fl_config.evaluation_metric,
            'participation_fee': fl_config.participation_fee,
            'minimum_contribution_value': fl_config.minimum_contribution_value,
            'created_at': fl_config.created_at,
        })

    return response




class FLConfigObject(BaseModel):
    batch_size: int
    test_batch_size: int
    learning_rate: float
    epochs: int
    evaluation_metric: int
    participation_fee: int
    minimum_contribution_value: int


@app.post("/fl-config")
def write_fl_config(config: FLConfigObject):
    fl_config = FLConfig()

    fl_config.batch_size = config.batch_size
    fl_config.test_batch_size = config.test_batch_size
    fl_config.evaluation_metric = config.evaluation_metric
    fl_config.learning_rate = config.learning_rate
    fl_config.epochs = config.epochs
    fl_config.participation_fee = config.participation_fee
    fl_config.minimum_contribution_value = config.minimum_contribution_value

    save_fl_config(fl_config)

    return {'success': True}
