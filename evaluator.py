from env_customizer import *

def evaluate_single_agents(env_class, student_customizers, num_episodes=100, seeds=None):
    results = {}
    if seeds is None:
        seeds = [i for i in range(num_episodes)]
        
    for name, customizer in student_customizers.items():
        env = CustomizedEnvironment(env=env_class, customizer=customizer, use_custom_reward=False)
        total_reward = 0
        for seed in seeds:
            obs = env.reset(seed=seed)
            done = False
            while not done:
                action = student_agents[student_name].compute_single_action(observation=obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
        average_reward = total_reward / num_episodes
        results[student_name] = average_reward
        env.close()
    return results

def evaluate_multi_agents(env_class, student_customizers, num_episodes=100, seeds=None):
    pass