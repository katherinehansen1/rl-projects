from datetime import datetime, timedelta
import statistics
from otree.api import *
import random
import os
import time
doc = """
"""
random.seed(1234)
os.environ['TZ'] = 'America/Los_Angeles'
#This function only exists un Unix systems only not on windows
time.tzset()
class C(BaseConstants):
    NAME_IN_URL = 'chapman'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 20
class Subsession(BaseSubsession):
    pass

def creating_session(subsession):
    initial_date = subsession.session.config['initial_date']
    initial_hour = subsession.session.config['initial_hour']
    period_length = subsession.session.config['PERIOD_LENGTH']
    format = "%Y-%m-%d %H:%M:%S"
    start = datetime.strptime(initial_date + ' ' + initial_hour, format)
    final = datetime.now()
    for player in subsession.get_players():
            for round in range(1, C.NUM_ROUNDS + 1): 
            # the second difference is the time that page 0 exists
                    player.start_time = start.timestamp()
                    first_dif = final + timedelta(minutes = period_length)*round
                    second_dif = start.timestamp() - datetime.now().timestamp()
                    player.in_round(round).participant.final_time = first_dif.timestamp() + second_dif 
                    player.in_round(round).final_time = first_dif.timestamp() + second_dif 
                    player.participant.expiry = final.timestamp() + player.start_time - datetime.now().timestamp()#span multiple pages. Including the time of the 
        
            if subsession.round_number == 1:  
                #TODO: make the code to read a csv file for the variables
                player.participant.budget = 1000
                for round in range(1, C.NUM_ROUNDS + 1):
                    # How many projects will the person receive this round
                    num_projects = random.randint(0, 2) #read.csv(num projec)
                    for i in range(1, num_projects + 1):
                        lower = random.randint(0, 7)
                        upper = lower + random.randint(1, 7)
                        payoff = random.randint(lower, upper)
                        final_round = random.randint(round, C.NUM_ROUNDS)
                        time_required = random.randint(1, 5)
                        Project.create(
                                project_id = str(round) + "-" + str(i),
                                player = player.in_round(round), 
                                payoff_lower = lower, 
                                payoff_upper = upper, 
                                payoff = payoff,
                                time_required_c = time_required,
                                time_required_var = time_required,
                                initial_round = round, 
                                budget_required = random.randint(10, 100),
                                final_round = final_round,
                                time_remaining = final_round - round,
                                invest = False
                        ) 
class Group(BaseGroup):
    pass

class Player(BasePlayer):
   final_time = models.FloatField()
   start_time = models.FloatField()
   invest = models.BooleanField(blank = True)
   budget = models.IntegerField()
   date = models.StringField()
   
class Project(ExtraModel):
    player = models.Link(Player)
    project_id = models.StringField()
    payoff_lower = models.StringField()
    payoff_upper = models.StringField()
    payoff = models.IntegerField()
    #Time required for building the project once choosen.
    time_required_c = models.IntegerField()
    time_required_var = models.IntegerField()
    #time remaining for choosing the project
    time_remaining = models.IntegerField()
    budget_required = models.IntegerField()
    initial_round = models.IntegerField()
    final_round = models.IntegerField()
    invest = models.BooleanField()
    round_selected = models.IntegerField()
    
# PAGES
@staticmethod
def get_timeout_seconds0(player):
    return player.start_time - datetime.now().timestamp()

class FrontPage(Page):

    timer_text = 'The activity will start in:'
    
    get_timeout_seconds= get_timeout_seconds0
    @staticmethod
    def is_displayed(player: Player):
        return player.start_time > datetime.now().timestamp()
        
# Creating session 
@staticmethod
def get_timeout_seconds1(player):
    return player.final_time - datetime.now().timestamp()

@staticmethod
def live_method(player: Player, data):
    # Get Project
    project = Project.filter(project_id = data['project'], player = player.in_round(int(data['project'][0])))[0] 
    new_budget = player.participant.budget - project.budget_required
    if new_budget >= 0 and project.invest == False:
        player.participant.budget = new_budget #this is only if the budget is accumulativo
        #player.participant.budget = new_budget
        project.invest = True
        project.round_selected = player.round_number
        player.invest = True
        return {player.id_in_group: {"new_budget": new_budget, "status": 1, "color": "green", "project_id": project.project_id}}
    else:
        return {player.id_in_group: {"error": 1, "project_id": project.project_id, "color": "grey"}}

class Investment(Page):
    
    live_method = live_method
    form_model = "player" 
    
    get_timeout_seconds = get_timeout_seconds1
 
    
    @staticmethod
    def vars_for_template(player):
        past_projects = []
        available_projects = []
        current_projects = []
        player.date = datetime.strftime(datetime.now(), "%b %d, %Y")
        for round in range(1, C.NUM_ROUNDS + 1):
            for p in Project.filter(player = player.in_round(round)):
                if ((p.initial_round <= player.round_number and p.final_round >= player.round_number) and p.invest == False):
                    available_projects.append(p)
                    
                if p.invest == True and p.time_required_var <= 0:
                    print(f'round_number: {player.round_number} final_round {p.final_round}')
                    past_projects.append(p)
                
                if p.time_required_var > 0 and p.invest == True:
                    current_projects.append(p)
       # Add current projects that pay to players payment 
        player.participant.payoff = sum([p.payoff for p in past_projects])
        
        # If the round is the one in which the project stop building, then give that money back
        player.participant.budget = player.participant.budget + sum([p.budget_required for p in past_projects if (p.round_selected + p.time_required_c +1) == player.round_number])
        
        return dict(
           periods_remain = C.NUM_ROUNDS - player.round_number, 
            available_projects = available_projects,
            current_projects = current_projects,
            past_projects = past_projects,
            #payoff_usd = cu(player.participant.payoff).to_real_world_currency(player.session),
            # If I want to show payoff in usd
            payoff_usd = player.participant.payoff
        )
    
    @staticmethod
    def is_displayed(player):

        for round in range(1, C.NUM_ROUNDS + 1):
            for p in Project.filter(player = player.in_round(round)):
                if ((p.initial_round <= player.round_number and p.final_round >= player.round_number) and p.invest == False):
                    p.time_remaining = p.final_round - player.round_number
                
                if p.time_required_var > 0 and p.invest == True:
                    if p.round_selected < player.round_number:
                        p.time_required_var = p.time_required_c + (p.round_selected -  player.round_number) + 1

        return player.final_time > datetime.now().timestamp()
    
page_sequence = [FrontPage,
                 Investment 
                 ]

def custom_export(players):
    '''
    This function exports the data from the projects
    '''
    # header row
    yield [
        'player','round', 'project_id','project_invest', 'project_lower_bound' , 'project_upper_bound', 'project_payoff', 'player_payoff', 'project_round_selected']
    for player in players:
        for project in Project.filter(player = player):
            yield [player.id_in_group, player.round_number, project.project_id, project.invest, project.payoff_lower, project.payoff_upper, project.payoff, player.participant.payoff,project.round_selected] 