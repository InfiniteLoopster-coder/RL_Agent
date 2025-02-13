# src/training/train_recommender.py
import os
import yaml
import numpy as np
import sys

# Append the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our custom modules.
from src.environments.rec_env import RecEnv
from src.models.dqn_agent import DQNAgent

def load_config(config_path="src/config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration parameters.
    config = load_config()
    env_params = config["env"]
    training_params = config["training"]

    # Create the environment.
    env = RecEnv(num_items=env_params["num_items"],
                 session_length=env_params["session_length"])
    
    # Optional: check that the environment follows the Gym API.
    check_env(env, warn=True)
    
    # Wrap the environment for Stable Baselines3.
    vec_env = DummyVecEnv([lambda: env])
    
    # Create and train the agent.
    agent = DQNAgent(vec_env, policy=training_params["policy"], verbose=training_params["verbose"])
    print("Starting training...")
    agent.train(total_timesteps=training_params["total_timesteps"])
    print("Training completed!")
    
    # Testing the trained agent.
    obs = vec_env.reset()
    total_reward = 0
    print("\n--- Testing the trained agent ---")
    for i in range(env.session_length):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        # Render current state.
        vec_env.envs[0].render()
        print(f"Step Reward: {reward[0]:.3f}\n")
        if done:
            break
    print("Total Reward for the session:", total_reward)

if __name__ == "__main__":
    main()
