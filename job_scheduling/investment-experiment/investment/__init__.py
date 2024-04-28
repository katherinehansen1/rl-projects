from datetime import datetime
from otree.api import *
import random
import os
import time
from . import job_env
import csv
import copy

doc = """
"""
random.seed(1234)
os.environ['TZ'] = 'America/Los_Angeles'
#This function only exists un Unix systems only not on windows
time.tzset()
class C(BaseConstants):
    NAME_IN_URL = 'chapman'
    PLAYERS_PER_GROUP = None
    #! The number of rounds can only be set here: check otree observation: https://www.otreehub.com/forum/193/
    NUM_ROUNDS = 100
    DEFAULT_START_DELAY = 5


def creating_session(subsession):
    if subsession.round_number == 1:
        session = subsession.session
        with open('./test_jobs.csv', mode='r') as file:
            csv_reader = csv.DictReader(file)
            session.n_days = session.config['max_rounds']
            session.jobs = [[] for _ in range(session.config['max_rounds']+1)]
            session.workers = session.config['workers']
            session.parts = session.config['parts']
            session.day_length = session.config['round_length']
            session.accumulate_time = session.config['accumulate_time']
            for row in csv_reader:
                progression = string_to_int_list(row['Progression'])
                job = job_env.Job(name=row['Name'],
                                  reqs=int(row['Workers']),
                                  parts=session.parts,
                                  complication=float(row['Complication Probability']),
                                  soft_deadline=int(row['Soft Deadline']),
                                  hard_deadline=int(row['Hard Deadline']),
                                  payment=int(row['Payment'])*session.config['pay_scale_factor'],
                                  progression=progression,
                                  late_penalty=session.config['late_penalty'],
                                  fail_penalty=session.config['fail_penalty']
                                )
                day=int(row['Day'])
                session.jobs[day-1].append(job)

        for player in subsession.get_players():
            player.participant.env = job_env.JobEnv(copy.deepcopy(session.jobs),
                                                    session.n_days,
                                                    session.workers,
                                                    worker_pay=session.config['worker_pay'])
        if 'initial_date' in session.config:
            initial_date = session.config['initial_date']
            initial_hour = session.config['initial_hour']
            format = "%Y-%m-%d %H:%M:%S"
            start = datetime.strptime(initial_date + ' ' + initial_hour, format)
            start = start.timestamp()
        else:
            start = time.time() + C.DEFAULT_START_DELAY
        session.start_time = start
        for player in subsession.get_players():
            player.participant.expiry = session.start_time + session.day_length

def get_seconds_until_start(player):
    session = player.session
    return session.start_time - time.time()

def get_period_time_seconds(player):
    participant = player.participant
    return participant.expiry - time.time()

def string_to_int_list(s):
    return list(map(int, s.split(',')))

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    offer_actions = models.StringField(blank=True)
    work_actions = models.StringField(blank=True)

    @property
    def player_id(self):
        return self.participant.id_in_session

class FrontPage(Page):
    timer_text = 'The experiment will start in:'
    get_timeout_seconds = get_seconds_until_start

    @staticmethod
    def is_displayed(player: Player):
        return get_seconds_until_start(player) > 0

class Investment(Page):
    form_model = "player"
    form_fields = ['offer_actions', 'work_actions']
    @staticmethod
    def js_vars(player):
        participant = player.participant
        env = participant.env
        jobs = env.jobs
        jobs = sorted(jobs, key=lambda x:(max(0,x.soft_deadline_remaining()), x.hard_deadline_remaining()))
        job_dict = [{'name':j.name,
                     'lengthRange':[j.lower_length(), j.upper_length()],
                     'pastSoft':j.is_late(),
                     'expectedLength':j.expected_length(),
                     'hardDeadline': j.hard_deadline_remaining(),
                     'softDeadline': j.soft_deadline_remaining()}
         for j in jobs]
        return dict(
            workers=player.session.workers,
            jobs=job_dict
        )

    @staticmethod
    def vars_for_template(player):
        participant = player.participant
        env = participant.env
        offers_strs = [job_env.common_job_str(job) for job in env.offers]

        job_strs = []
        ended_strs = []
        danger_bools = []
        if env.history.history:
            hist = env.history.history[-1]
            old_jobs = hist.jobs
            old_job_actions = hist.job_actions
            for job, worked in zip(old_jobs, old_job_actions):
                if not job.is_ended():
                    s = job_env.common_job_str(job)
                    parts_completed_this_day = job.last_progress() * worked
                    (cur, last) = job_env.progress_str(job, parts_completed_this_day)
                    s['progress_cur'] = cur
                    s['progress_last'] = last
                    s['status'] = ''
                    s['danger'] = job.is_late() or job.hard_deadline_remaining()==1
                    job_strs.append((max(0,job.soft_deadline_remaining()), job.hard_deadline_remaining(), s))
            new_jobs = hist.get_taken_jobs()
            for job in new_jobs:
                s = job_env.common_job_str(job)
                (cur, last) = job_env.progress_str(job, 0)
                s['progress_cur'] = cur
                s['progress_last'] = last
                s['status'] = "New"
                s['danger'] = job.is_late() or job.hard_deadline_remaining()==1
                job_strs.append((max(0,job.soft_deadline_remaining()), job.hard_deadline_remaining(), s))
            job_strs = sorted(job_strs, key=lambda x:(x[0], x[1]))
            job_strs = [x[2] for x in job_strs]

            ended_jobs = hist.ended
            ended_actions = hist.ended_actions
            for job, worked in zip(ended_jobs, ended_actions):
                s = job_env.common_job_str(job)
                parts_completed_this_day = job.last_progress() * worked
                (cur, last) = job_env.progress_str(job, parts_completed_this_day)
                s['progress_cur'] = cur
                s['progress_last'] = last
                s['status'] = 'Failed' if job.failed else 'Completed'
                ended_strs.append(s)
        return dict(
            offer_strs = offers_strs,
            job_strs = job_strs,
            ended_strs = ended_strs,
            danger_bools = danger_bools,
            payoff_usd = f"${participant.env.total_payment/100:.2f}",
            workers = player.session.workers,
            allow_submit = player.session.config['allow_submit']
        )
    get_timeout_seconds = get_period_time_seconds

    @staticmethod
    def before_next_page(player, timeout_happened):
        participant = player.participant
        taken_job_names = set(player.offer_actions.split(','))
        offer_actions = []
        for job in participant.env.offers:
            if job.name in taken_job_names:
                offer_actions.append(1)
            else:
                offer_actions.append(0)
        worked_job_names = set(player.work_actions.split(','))
        work_actions = []
        for job in participant.env.jobs:
            if job.name in worked_job_names:
                work_actions.append(1)
            else:
                work_actions.append(0)
        participant.env.step(offer_actions, work_actions)
        participant.payoff = participant.env.total_payment
        if player.session.accumulate_time:
            participant.expiry = max(time.time(), participant.expiry) + player.session.day_length
        else:
            participant.expiry = time.time() + player.session.day_length

    @staticmethod
    def is_displayed(player):
        return player.round_number <= player.session.config['max_rounds']


page_sequence = [FrontPage,Investment]

def custom_export(players):
    yield [
        'SubjectID','Day', 'Job Name','Action Type', 'Accepted/Worked On',
        'Parts Completed', 'Soft Deadline Remaining', 'Hard Deadline Remaining', 'Current Payment', 'Payment Received']
    for player in players:
        participant = player.participant
        env = participant.vars['env']
        if len(env.history.history) > player.round_number-1:
            hist = env.history.history[player.round_number-1]
            for job, action in zip(hist.offers, hist.offer_actions):
                yield [player.id_in_group, player.round_number, job.name, 'Offer', action,
                       '','','','','']
            for job, action in zip(hist.jobs, hist.job_actions):
                received = '' if not job.is_ended() else job.final_payment
                yield [player.id_in_group, player.round_number, job.name, 'Work', action,
                       job.parts_completed, job.soft_deadline_remaining(), job.hard_deadline_remaining(), job.payment_current(), received]
