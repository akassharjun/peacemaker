import json

import pika
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pika import DeliveryMode

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
        'peacemaker_exchange', 'peacemaker_key', 'queue:group',
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
    # training_rounds.
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
                payout = payout_info.payout
                contribution = payout_info.contribution_measure

        filtered_training_rounds.append(
            {
                'date': tround.created_at,
                'training_round_id': str(tround.id),
                'participation_fee': tround.fl_config.participation_fee,
                'payout': payout,
                'contribution': contribution
            }
        )

    return filtered_training_rounds
