from db.dao import *

connect('peacemaker')


def get_training_rounds():
    return TrainingRound.objects()


def get_global_models():
    return GlobalModel.objects()


def get_fl_configs():
    return FLConfig.objects()


def save_fl_config(fl_config):
    fl_config.save()
