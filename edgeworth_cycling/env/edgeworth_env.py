'''
This code is a slightly simplified version of the environment found in the paper:
Oligopoly competition in fixed cost environments
https://www.sciencedirect.com/science/article/pii/S0167718703001577
'''

import numpy as np
import gymnasium as gym
from functools import partial
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

def calculate_sales(offers, intercept, slope):
    total_sold = 0
    sales = {}

    offers = sorted([(offer[0], offer[1], aid) for aid, offer in offers.items()])
    i = 0
    while i < len(offers):
        # Find all offers with the same price at the current index
        current_price = offers[i][0]
        next_i = i+1
        while next_i < len(offers) and offers[next_i][0] == current_price:
            next_i += 1
        n_equal = next_i - i

        # Calculate remaining demand at this price
        demand = (intercept - current_price) / slope - total_sold
        demand = max(0,demand)
        for k in range(i, next_i):
            demand_share = demand / n_equal

            quantity = offers[k][1]
            firm_id = offers[k][2]
            actual_sold = min(quantity, demand_share)
            sales[firm_id] = actual_sold
            total_sold += actual_sold
            demand -= actual_sold

            n_equal -= 1
        i = next_i
    return sales


class OligopolyMarket(MultiAgentEnv):
    def __init__(self, seed=None):
        super().__init__()
        self.n_firms = 5
        self._agent_ids = [f'firm_{i}' for i in range(self.n_firms)]

        self.history_length = 5
        self.price_history = []
        self.intercept = 17
        self.slope = 0.01
        self.n_periods = 80

        # Actions are tuples of (price, quantity) for each firm
        self.ind_action_space = spaces.Tuple((
            spaces.Box(low=0,high=self.intercept),
            spaces.Box(low=0,high=300)
        ))

        self.ind_observation_space = spaces.Box(
            low=0,
            high=self.intercept,
            shape=(self.history_length,)
        )
        self.action_space = gym.spaces.Dict(self.make_agent_dictionary(self.ind_action_space))
        self.observation_space = gym.spaces.Dict(self.make_agent_dictionary(self.ind_observation_space))

        self.costs = [4.0]*100 + [4.5]*100 + [5.0]*100
        self.sales_calc = partial(calculate_sales, intercept=self.intercept, slope=self.slope)

    def make_agent_dictionary(self, res):
        return {agent_id: res for agent_id in self._agent_ids}

    def get_observation(self):
        price_history = list(reversed(self.price_history[-self.history_length:]))
        return self.make_agent_dictionary(price_history)

    def reset(self, seed=None, options=None):
        self.price_history = [0]*self.history_length
        self.period = 0
        return self.get_observation(), {}

    def step(self, actions):
        self.period += 1
        prices = {aid: np.round(action[0][0],2) for aid,action in actions.items()}
        quantities = {aid: action[1][0] for aid,action in actions.items()}
        offers = {aid: (prices[aid],quantities[aid]) for aid in self._agent_ids}
        sales = self.sales_calc(offers)
        profits = {}
        for aid in self._agent_ids:
            q = sales[aid]
            p = prices[aid]
            costs = sum(self.costs[:int(q)])
            # Add costs in event they sold fraction of a unit
            if q < len(self.costs):
                costs += (q - int(q))*self.costs[int(q)]
            profits[aid] = p*q - costs
        tot_sales = sum(sales.values())
        if tot_sales > 0:
            avg_price = sum(sales[aid]*prices[aid] for aid in sales.keys())/sum(sales.values())
        else:
            avg_price = self.intercept
        self.price_history.append(avg_price)

        rewards = self.get_rewards(profits)
        done = self.period == self.n_periods
        dones = self.make_agent_dictionary(done)
        dones['__all__'] = done
        truncated = dones
        info = {}
        return self.get_observation(), rewards, dones, truncated, info

    def get_evaluation_reward(self, profits):
        return profits

    def get_rewards(self, profits):
        return self.get_evaluation_reward(profits)
