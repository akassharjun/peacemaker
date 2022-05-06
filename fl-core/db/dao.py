from datetime import datetime
from enum import Enum
from mongoengine import *


class EvaluationMetric(Enum):
    ACCURACY = 1
    PRECISION = 2
    F1_SCORE = 3
    RECALL = 4


class Organization(Document):
    name = StringField()


class FLConfig(Document):
    use_cuda = BooleanField()
    batch_size = IntField()
    test_batch_size = IntField()
    learning_rate = FloatField()
    evaluation_metric = EnumField(enum=EvaluationMetric)
    log_interval = IntField()
    epochs = IntField()
    participation_fee = FloatField()
    created_at = DateTimeField(default=datetime.now())

    @queryset_manager
    def objects(self, query_set):
        return query_set.order_by('-created_at')


class PayoutReport(EmbeddedDocument):
    epoch = IntField()
    organization = ReferenceField(Organization)
    contribution_measure = FloatField()
    payout = FloatField()


class CommunicationTime(EmbeddedDocument):
    epoch = IntField()
    time_taken = FloatField()


class TrainingRoundStatus(Enum):
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3


class TrainingRound(Document):
    budget = FloatField()
    status = EnumField(TrainingRoundStatus)
    organizations = ListField(ReferenceField(Organization))
    payout_report = ListField(EmbeddedDocumentField(PayoutReport))
    communication_time = ListField(EmbeddedDocumentField(CommunicationTime))
    fl_config = ReferenceField(FLConfig)
    created_at = DateTimeField(default=datetime.now())


class ModelReport(EmbeddedDocument):
    accuracy = FloatField()
    precision = FloatField()
    recall = FloatField()
    f1_score = FloatField()


class GlobalModel(Document):
    model_id = UUIDField()
    training_round = ReferenceField(TrainingRound)
    model_report = EmbeddedDocumentField(ModelReport)
    organizations = ListField(ReferenceField(Organization))
    created_at = DateTimeField(default=datetime.now())

    @queryset_manager
    def objects(self, query_set):
        return query_set.order_by('-created_at')
