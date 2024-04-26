from otree.api import *


doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'summary'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pass


# PAGES
class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        return dict(
            uds_payoff = cu(player.participant.payoff_plus_participation_fee()).to_real_world_currency(player.session)
        )


page_sequence = [
    Results]
