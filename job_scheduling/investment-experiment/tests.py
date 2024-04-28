from otree.api import Currency as c, currency_range, expect, Bot, Submission
import random
from . import *

class MyBot(Bot):
    def play_round(self):
        round_number = self.subsession.round_number

        # Calculate the maximum number of buttons available based on your logic
        max_buttons_available = self.player.calculate_max_buttons_available()

        # Construct the button IDs based on the format
        button_ids = [f'button{round_number}_{button_id}' for button_id in range(1, max_buttons_available + 1)]

        # Click each button with a 25% chance
        for button_id in button_ids:
            if random.random() <= 0.25:
                self.player.click(id=button_id)

        self.player.execute_script('checkAllCheckboxes();')

        yield Submission(self.player, check_html=False)