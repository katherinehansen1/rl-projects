import numpy as np
from scipy.stats import binom
import copy

class Job:
    def __init__(self, name, reqs, parts, complication, soft_deadline, hard_deadline, payment, progression, late_penalty=0.15, fail_penalty=0.2):
        self.name = name
        self.n_workers = reqs
        self.parts = parts
        self.parts_completed = 0
        self.days_worked = 0
        self.days_passed = 0
        self.complication_probability = complication
        self.soft_deadline = soft_deadline
        self.hard_deadline = hard_deadline
        self.payment = payment
        self.final_payment = -1
        self._progression=progression
        self.completed = False
        self.failed = False
        self._late_penalty = late_penalty
        self._fail_penalty = fail_penalty

    def parts_remaining(self):
        return self.parts - self.parts_completed

    def soft_deadline_remaining(self):
        return self.soft_deadline - self.days_passed

    def is_late(self):
        return self.soft_deadline_remaining() <= 0

    def hard_deadline_remaining(self):
        return self.hard_deadline - self.days_passed

    def payment_current(self):
        if self.hard_deadline_remaining() <= 0:
            return int(np.round(-self._fail_penalty * self.payment))
        elif self.soft_deadline_remaining() <= 0:
            n_days_past = np.abs(self.soft_deadline_remaining() - 1)
            return int(np.round((1 - self._late_penalty*n_days_past) * self.payment))
        else:
            return self.payment

    def expected_length(self, current=True):
        parts = self.parts if not current else self.parts_remaining()
        if parts == 0:
            return 0
        return 1 + (parts-1)*self.complication_probability

    def upper_length(self, current=True):
        parts = self.parts if not current else self.parts_remaining()
        if parts == 0:
            return 0
        return int(1 + binom(parts-1,self.complication_probability).ppf(0.95))

    def lower_length(self, current=True):
        parts = self.parts if not current else self.parts_remaining()
        if parts == 0:
            return 0
        return int(1 + binom(parts-1,self.complication_probability).ppf(0.05))

    def return_rate(self, current=True, omniscient=False):
        parts = self.parts if not current else self.parts_remaining()
        payment = self.payment if not current else self.payment_current()
        if omniscient:
            worked = self.days_worked if current else 0
            length = len(self._progression) - worked
        else:
            length = self.expected_length(current)
        if parts == 0:
            return 0
        return payment / (self.n_workers * length)

    def advance_day(self):
        if not self.is_ended():
            self.days_passed += 1
        if self.hard_deadline_remaining() <= 0 and not self.completed:
            self.failed = True
            self.on_time = False
            self.final_payment = self.payment_current()

    def work(self):
        self.parts_completed += self._progression[self.days_worked]
        self.days_worked += 1
        if self.parts_remaining() == 0:
            self.completed = True
            self.on_time = self.soft_deadline_remaining() > 0
            self.final_payment = self.payment_current()

    def is_ended(self):
        return self.failed or self.completed

    def last_progress(self):
        if self.days_worked == 0:
            return 0
        else:
            return self._progression[self.days_worked - 1]

    def __str__(self):
        return (f'{self.name}: ' +
                f'{self.n_workers} Workers, ' +
                f'{self.parts_remaining()} Parts, ' +
                f'{self.complication_probability*100:2.0f}%, ' +
                f'{self.soft_deadline_remaining():2}/{self.hard_deadline_remaining():2} Deadlines, ' +
                f'{self.payment_current()} Cents')


def progress_str(job, completed_this_day, include_zero=False):
    perc_comp = job.parts_completed / job.parts * 100
    added_perc_comp = completed_this_day / job.parts * 100
    cur = f"{perc_comp:.0f}%"
    if include_zero or completed_this_day > 0:
        last = f" (+{added_perc_comp:.0f}%)"
    else:
        last = ""
    return cur, last


def common_job_str(job, current=True):
    s_deadline = job.soft_deadline_remaining() if current else job.soft_deadline
    h_deadline = job.hard_deadline_remaining() if current else job.hard_deadline
    pay = job.payment_current() if current else job.payment
    if h_deadline <= 0:
        payment_str = f""
        penalty_str = f"-${-pay/100:.2f}"
    elif s_deadline <= 0 and not job.is_ended():
        payment_str = f"${pay/100:.2f}"
        penalty_str = f" (-${(job.payment-pay)/100:.2f})"
    else:
        payment_str = f"${pay/100:.2f}"
        penalty_str = ""
    if job.is_ended():
        if h_deadline <= 0:
            deadline_str = "Incomplete"
        elif s_deadline == 0:
            deadline_str = f"1 Day Late"
        elif s_deadline < 0:
            deadline_str = f"{1-s_deadline} Days Late"
        else:
            deadline_str = "On Time"
    else:
        if h_deadline == 1:
            deadline_str = f"Last Day"
        elif s_deadline <= 0:
            deadline_str = f"0/{h_deadline} Days"
        else:
            deadline_str = f"{max(0,s_deadline):2}/{h_deadline} Days"
    length_str = f"{job.expected_length(current):2.1f} Days [{job.lower_length(current)}--{job.upper_length(current)}]"
    rate_str = f"{job.return_rate(current):2.1f}"
    return {'name':job.name,
            'workers':job.n_workers,
            'payment':payment_str,
            'penalty':penalty_str,
            'deadlines':deadline_str,
            'lengths':length_str,
            'rate':rate_str}


class DayHistory:
    def __init__(self, day, offers, offer_actions, jobs_before, job_actions, n_workers):
        self.day = day
        self.offers = offers
        self.offer_actions = offer_actions
        self.jobs = jobs_before
        self.job_actions = job_actions
        self.ended = [job for job in jobs_before if job.is_ended()]
        self.ended_actions = [job_actions[i] for i in range(len(job_actions)) if self.jobs[i] in self.ended]
        self._n_workers_assigned = None
        self._utilization = None
        self._active_worker_rate = [None,None]
        self._worker_rate = [None, None]
        self._average_length = [None, None]
        self._average_workers = None
        self._expected_commitment = [None, None]
        self._value = None
        self.n_workers = n_workers

    def get_taken_jobs(self):
        return [self.offers[i] for i in range(len(self.offers)) if self.offer_actions[i] == 1]

    def get_untaken_jobs(self):
        return [self.offers[i] for i in range(len(self.offers)) if self.offer_actions[i] != 1]

    def get_worked_jobs(self):
        return [self.jobs[i] for i in range(len(self.jobs)) if self.job_actions[i] == 1]

    def get_n_workers_assigned(self):
        if self._n_workers_assigned is None:
            self._n_workers_assigned = sum(job.n_workers for job in self.get_worked_jobs())
        return self._n_workers_assigned

    def get_utilization(self):
        return self.get_n_workers_assigned()/self.n_workers

    def get_hindsight_worker_rate(self):
        tot_return = sum(max(0,job.final_payment) for job in self.get_worked_jobs())
        return tot_return/self.n_workers

    def get_worker_rate(self, current=False):
        if self._worker_rate[int(current)] is None:
            tot_return = sum(job.n_workers*job.return_rate(current) for job in self.get_worked_jobs())
            self._worker_rate[int(current)] = tot_return/self.n_workers
        return self._worker_rate[int(current)]

    def get_active_worker_rate(self, current=False):
        if self.get_n_workers_assigned() == 0:
            return 0
        return self.get_worker_rate(current)*self.n_workers/self.get_n_workers_assigned()

    def get_average_length(self, current=True):
        if len(self.jobs) == 0:
            return 0
        if self._average_length[int(current)] is None:
            self._average_length[int(current)] = np.mean([job.expected_length(current) for job in self.jobs])
        return self._average_length[int(current)]

    def get_average_workers(self):
        if len(self.jobs) == 0:
            return 0
        if self._average_workers is None:
            self._average_workers = np.mean([job.n_workers for job in self.jobs])
        return self._average_workers

    def get_expected_commitment(self, current=True):
        if self._expected_commitment[int(current)] is None:
            self._expected_commitment[int(current)] = sum(job.expected_length(current)*job.n_workers for job in self.jobs)
        return self._expected_commitment[int(current)]

    def get_payment(self):
        if self._value is None:
            self._value = sum(job.final_payment for job in self.ended)
        return self._value

def flatten(lsts):
    return [item for sublist in lsts for item in sublist]

class EnvHistory:
    def __init__(self, n_days, n_workers):
        self.history = []
        self.n_days = n_days
        self.n_workers = n_workers
        self._reset_cache()

    def _reset_cache(self):
        self._n_taken = None
        self._n_workers_assigned = None
        self._total_payments = None
        self._n_ended = None

    def record(self, day_hist):
        self.history.append(day_hist)
        self._reset_cache()

    def get_offered(self, flattened=False):
        lsts = [day_hist.offers for day_hist in self.history]
        return flatten(lsts) if flattened else lsts

    def get_taken(self, flattened=False):
        lsts = [day_hist.get_taken_jobs() for day_hist in self.history]
        return flatten(lsts) if flattened else lsts

    def get_untaken(self, flattened=False):
        lsts = [day_hist.get_untaken_jobs() for day_hist in self.history]
        return flatten(lsts) if flattened else lsts

    def get_ended(self, flattened=False):
        lsts = [day_hist.ended for day_hist in self.history]
        return flatten(lsts) if flattened else lsts

    def get_n_taken(self):
        if self._n_taken is None:
            self._n_taken = sum(len(day_taken) for day_taken in self.get_taken())
        return self._n_taken

    def get_n_ended(self):
        if self._n_ended is None:
            self._n_ended = sum(len(day_ended) for day_ended in self.get_ended())
        return self._n_ended

    def acceptance_rate(self):
        n_untaken = sum(len(day_untaken) for day_untaken in self.get_untaken())
        return self.get_n_taken() / (self.get_n_taken() + n_untaken)

    def completion_rate(self):
        n_completed = sum(job.completed for job in self.get_ended(flattened=True))
        return n_completed/self.get_n_ended()

    def on_time_rate(self):
        n_ontime = sum(job.on_time for job in self.get_ended(flattened=True))
        return n_ontime / self.get_n_ended()

    def avg_accepted_length(self):
        if self.get_n_taken == 0:
            return 0
        tot_lengths = sum(job.expected_length(current=False) for job in self.get_taken(flattened=True))
        return tot_lengths / self.get_n_taken()

    def avg_accepted_workers(self):
        if self.get_n_taken == 0:
            return 0
        tot_workers = sum(job.n_workers for job in self.get_taken(flattened=True))
        return tot_workers / self.get_n_taken()

    def avg_accepted_rate(self):
        if self.get_n_taken == 0:
            return 0
        tot_rates = sum(job.return_rate(current=False) for job in self.get_taken(flattened=True))
        return tot_rates / self.get_n_taken()

    def get_n_workers_assigned(self):
        if self._n_workers_assigned is None:
            self._n_workers_assigned = sum(day_hist.get_n_workers_assigned() for day_hist in self.history)
        return self._n_workers_assigned

    def utilization(self):
        return self.get_n_workers_assigned() / (self.n_workers*len(self.history))

    def total_payment(self, day=None):
        if self._total_payments is None:
            self._total_payments = np.cumsum([day_hist.get_payment() for day_hist in self.history])
        if not day:
            day = len(self._total_payments)
        return self._total_payments[day-1]

    def utilized_rate(self):
        return self.total_payment() / self.get_n_workers_assigned()

    def rate(self):
        return self.total_payment() / (self.n_workers*len(self.history))


class JobEnv():
    def __init__(self, all_jobs, n_days, n_workers, worker_pay=0):
        self.n_days = n_days
        self.n_workers = n_workers
        self.jobs = []
        self.worker_pay=worker_pay
        self.all_jobs=all_jobs
        self.current_day=0
        self.total_payment = 0
        self.history = EnvHistory(self.n_days, self.n_workers)
        self.offers = self.all_jobs[self.current_day]

    def step(self, job_acceptances, work_actions):
        if self.current_day >= self.n_days:
            return

        used = 0
        realized_actions = copy.copy(work_actions)
        active_jobs = self.jobs.copy() # Copy so removing jobs from self.jobs doesn't mess up iterating
        for i,job in enumerate(active_jobs):
            if work_actions[i] ==1:
                if used + job.n_workers <= self.n_workers:
                    used += job.n_workers
                    self.total_payment += self.process_job(job)
                else:
                    realized_actions[i] = -1

        self.total_payment += self.advance_day(active_jobs)
        self.total_payment -= self.worker_pay
        realized_acceptances = copy.copy(job_acceptances)
        for i, job in enumerate(self.offers):
            if job_acceptances[i] == 1:
                self.take_job(job)

        hist = DayHistory(self.current_day,
                          copy.deepcopy(self.offers),
                          realized_acceptances,
                          copy.deepcopy(active_jobs),
                          realized_actions,
                          self.n_workers)
        self.history.record(hist)
        if self.current_day < self.n_days:
            self.offers = self.all_jobs[self.current_day]
        else:
            self.offers = []


    def process_job(self, job) -> float:
        job.work()
        if job.completed:
            self.complete_job(job)
            return job.final_payment
        else:
            return 0


    def advance_day(self, active_jobs):
        incurred_penalties = 0
        self.current_day += 1
        for job in active_jobs:
            if not job.is_ended():
                job.advance_day()
            if job.failed:
                self.complete_job(job)
                incurred_penalties += job.final_payment
        return incurred_penalties


    def complete_job(self, job):
        job.final_payment = job.payment_current()
        self.jobs.remove(job)


    def take_job(self, job):
        self.jobs.append(job)
