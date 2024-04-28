import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv



class EmptyEnvCustomizer:
    def __init__(self):
        self.observation_space = None
        self.action_space = None

    def reward(self, env, rew):
        return rew

    # If you change the observation you must define self.observation_space
    # and convert the default obs to your custom one in this function
    def transform_observation(self, env, obs):
        return obs

    # If you change the action you must define self.action_space
    # and convert the default action to your custom one in this function
    def transform_action(self, env, action):
        return action


class CustomizedEnvironment(gym.Env):
    def __init__(self, env, customizer=None, use_custom_reward=True, seed=None):
        self.env = env
        self.customizer = customizer if customizer else EmptyEnvCustomizer()
        self.use_custom_reward = use_custom_reward
        self.observation_space = self.customizer.observation_space or env.observation_space
        self.action_space = self.customizer.action_space or env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.customizer.transform_observation(self.env, obs)
    
    def step(self, action_in):
        action = self.customizer.transform_action(self.env, action_in)
        obs, rew, done, trunc, info = self.env.step(action)
        obs = self.customizer.transform_observation(self.env, obs)
        if self.use_custom_reward:
            rew = self.customizer.reward(self.env, rew)
        return obs, rew, done, trunc, info

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()
    
    
class CustomizedMultiEnvironment(MultiAgentEnv):
    def __init__(self, env, customizers=None, use_custom_reward=True, seed=None):
        self.env = env
        self.customizers = {aid: customizers.get(aid, EmptyEnvCustomizer) for aid in env._agent_ids}
        self.use_custom_reward = use_custom_reward
        self.observation_space = {aid: cust.observation_space or env.observation_space for aid, cust in self.customizers.items()}
        self.action_space = {aid: cust.action_space or env.action_space for aid, cust in self.customizers.items()}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return {aid: cust.transform_observation(self.env, obs[aid]) for aid, cust in self.customizers.items()}
    
    def step(self, actions_in):
        actions = {aid: cust.transform_action(self.env, actions_in[aid]) for aid, cust in self.customizers.items()}
        obs, rew, done, trunc, info = self.env.step(actions)
        obs = {aid: cust.transform_observation(self.env, obs[aid]) for aid, cust in self.customizers.items()}
        if self.use_custom_reward:
            rew = {aid: cust.reward(self.env, rew[aid]) for aid, cust in self.customizers.items()}
        return obs, rew, done, trunc, info

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()