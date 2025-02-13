# src/environments/rec_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class RecEnv(gym.Env):
    """
    A simplified recommendation environment using Gymnasium.
    
    - There are 'num_items' items.
    - Each episode simulates a user session.
    - The user has a hidden preference distribution (sampled from a Dirichlet distribution).
    - At each step, the agent recommends an item. The reward is:
        reward = (user_preference for the item) / (number of times this item has been recommended)
      This simulates diminishing returns for repeated recommendations.
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, num_items=10, session_length=20):
        super(RecEnv, self).__init__()
        self.num_items = num_items
        self.session_length = session_length
        
        # Action space: integer index for each item.
        self.action_space = spaces.Discrete(num_items)
        
        # Observation space: counts for each item.
        self.observation_space = spaces.Box(
            low=0, high=session_length, shape=(num_items,), dtype=np.int32
        )
        self.reset()

    def reset(self, **kwargs):
        self.t = 0  # time step counter
        self.counts = np.zeros(self.num_items, dtype=np.int32)  # recommendation counts
        # Hidden user preference vector (latent engagement probabilities).
        self.user_pref = np.random.dirichlet(np.ones(self.num_items))
        # Gymnasium reset should return (observation, info)
        return self.counts.copy(), {}

    def step(self, action):
        # Update recommendation count for the chosen item.
        self.counts[action] += 1
        # Compute reward: higher if the user prefers the item and if it hasn't been over-recommended.
        reward = self.user_pref[action] / self.counts[action]
        self.t += 1
        done = self.t >= self.session_length
        truncated = False  # If you don't plan on truncating episodes manually, set to False.
        # Return (observation, reward, done, truncated, info)
        return self.counts.copy(), reward, done, truncated, {}

    def render(self, mode="human", close=False):
        print(f"Time Step: {self.t}")
        print("Recommendation Counts:", self.counts)
        print("User Preference:", np.round(self.user_pref, 3))
