from stable_baselines3 import DQN

class DQNAgent:
    def __init__(self, env, policy="MlpPolicy", verbose=1):
        self.env = env
        self.policy = policy
        self.verbose = verbose
        self.model = DQN(policy, env, verbose=verbose)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = DQN.load(path, env=self.env)
