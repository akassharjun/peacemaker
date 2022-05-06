from db.dao import *

connect('peacemaker')

def get_training_rounds():
    return TrainingRound.objects()
