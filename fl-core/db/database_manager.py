from db.dao import *

connect('peacemaker')


def get_organisations():
    return Organization.objects()


def get_fl_config():
    return FLConfig.objects()[0]


def get_organization(name):
    if len(Organization.objects(name=name)) != 0:
        return Organization.objects(name=name).first()

    return None


def prepare_payout(name, payout, contribution_measure):
    payout_report = PayoutReport()
    payout_report.payout = payout
    payout_report.contribution_measure = contribution_measure
    org = get_organization(name)

    payout_report.organization = org.save().to_dbref()

    return payout_report


def prepare_comm_time(epoch, time_taken):
    comm = CommunicationTime()
    comm.epoch = epoch
    comm.time_taken = time_taken

    return comm


def create_training_round(fl_config, organizations):
    training_round = TrainingRound()

    orgs = []
    for org in organizations:
        org.save()
        orgs.append(org.to_dbref())

    training_round.organizations = orgs
    training_round.fl_config = fl_config.save().to_dbref()
    training_round.save()

    return training_round.id


def get_training_round(id):
    return TrainingRound.objects(id=id).first()


def update_training_round(training_round_id, budget, payout_report, communication_time):
    training_round = get_training_round(training_round_id)

    payouts = []
    for org, value in payout_report.items():
        payout = prepare_payout(org, value['payout'], value['contribution_measure'])
        payouts.append(payout)

    communication_report = []
    for epoch, time in communication_time.items():
        comm = prepare_comm_time(epoch, time)
        communication_report.append(comm)

    TrainingRound.objects(id=training_round.id).update(communication_time=communication_report, payout_report=payouts,
                                                       budget=budget)
    training_round.reload()

    return training_round


def save_global_model(model_id, model_report, training_round):
    global_model = GlobalModel()
    global_model.model_id = model_id

    model_result = ModelReport()

    model_result.accuracy = model_report['accuracy']
    model_result.precision = model_report['precision']
    model_result.f1_score = model_report['f1_score']
    model_result.recall = model_report['recall']

    global_model.model_report = model_result
    global_model.training_round = training_round.to_dbref()
    global_model.save()


def get_global_model():
    if len(GlobalModel.objects()) != 0:
        return GlobalModel.objects()[0]

    return None
