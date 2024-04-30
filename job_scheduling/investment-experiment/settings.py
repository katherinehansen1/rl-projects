from os import environ
import time
from datetime import datetime, timedelta

SESSION_CONFIGS = [
     dict(
         name='inv_experiment',
         app_sequence=[
          'investment',
          'summary'
          ],
         num_demo_participants=6,
#         initial_date = datetime.strftime(datetime.now(), "%Y-%m-%d"), 
#         initial_hour = datetime.strftime(datetime.now() +timedelta(hours = 0) +timedelta(seconds = 30), "%H:%M:%S"),
         max_rounds = 100,
         round_length = 45,
         parts = 10,
         workers = 10,
         late_penalty = 0.15,
         fail_penalty = 0.2,
         allow_submit = True,
         accumulate_time = True,
         worker_pay = 0,
         pay_scale_factor = 1
     ),
]
ROOMS = [
    dict(
        name='econ_lab',
        display_name='Experimental Economics Lab',
        participant_label_file='_rooms/econ_lab.txt'
    )
    ]
# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=0.01,
    participation_fee=7.00,
    doc=""
)

use_browser_bots = True

PARTICIPANT_FIELDS = ['env', 'expiry']
SESSION_FIELDS = ['jobs', 'start_time', 'workers', 'parts', 'day_length', 'accumulate_time']

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD ='esi123'

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '1081876273808'

DEBUG=False
