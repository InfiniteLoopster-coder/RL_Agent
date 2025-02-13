# tests/test_rec_env.py
import pytest
import numpy as np
from src.environments.rec_env import RecEnv

def test_reset():
    env = RecEnv(num_items=5, session_length=10)
    obs = env.reset()
    # Initial counts should be zeros.
    assert np.all(obs == 0), "Initial observation should be a zero vector."

def test_step():
    env = RecEnv(num_items=5, session_length=10)
    env.reset()
    # Take an action (e.g., recommend item 0) and check the reward.
    obs, reward, done, info = env.step(0)
    # Check that count for item 0 increased.
    assert obs[0] == 1, "Item 0 count should be 1 after one recommendation."
    # Reward should be positive.
    assert reward > 0, "Reward should be positive."
    # Ensure done flag is boolean.
    assert isinstance(done, bool), "Done flag should be a boolean."

if __name__ == "__main__":
    pytest.main()
