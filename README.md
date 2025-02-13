# RL Recommendation System

This project implements a personalized recommendation system using Reinforcement Learning (RL). The system simulates user sessions with a custom Gym environment and trains a DQN agent (via Stable Baselines3) to learn a recommendation policy.

## Folder Structure

- **data/**: Raw and processed datasets (if any).
- **notebooks/**: Jupyter notebooks for exploration and visualization.
- **src/**: Source code:
  - **config/**: Configuration files.
  - **environments/**: Custom Gym environment (`rec_env.py`).
  - **models/**: RL agent wrapper (`dqn_agent.py`).
  - **training/**: Training scripts.
  - **utils/**: Utility functions (e.g., visualization).
- **tests/**: Unit tests.
- **requirements.txt**: Python dependencies.
- **README.md**: Project documentation.

## Setup Instructions

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
