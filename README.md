# RL Recommendation System

This project implements a personalized recommendation system using Reinforcement Learning (RL). The system simulates user sessions with a custom Gym environment and trains a DQN agent (via Stable Baselines3) to learn a recommendation policy.
![Screenshot 2025-02-13 182527](https://github.com/user-attachments/assets/e864ac41-8246-4b41-b15e-2f9b97f78f1a)


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
3. Run the Script from the Project Root
   ```bash
   python src/training/train_recommender.py



Project Overview:

This project builds a state-of-the-art personalized recommendation system using reinforcement learning (RL) techniques. The core idea is to enable an RL agent to learn optimal recommendation policies by interacting with a simulated user environment that mimics real-world behavior. The system is designed to dynamically adapt content suggestions based on individual user preferences and historical interactions, ensuring that recommendations remain engaging and relevant over time.

Key Components:

Custom Gymnasium Environment:
The environment simulates user sessions where each user is assigned a hidden preference vector generated from a Dirichlet distribution. This vector represents the probability of the user engaging with each of a set of items (e.g., movies, products, music). The environment maintains a count vector to track how many times each item is recommended. The reward function is designed to give a higher reward for recommending items that align with the user's preferences, while incorporating diminishing returns for repeated recommendations of the same item.

Reinforcement Learning Agent:
A Deep Q-Network (DQN) agent, implemented using Stable Baselines3, is used to navigate the recommendation space. The agent’s policy is learned through interactions within the environment, balancing exploration of new items and exploitation of known favorites. The RL approach enables the agent to optimize for both immediate rewards (e.g., click-through rates) and long-term user engagement.

Modular Design and Scalability:
The project is structured into clearly defined modules, including environment simulation, model definition, training routines, and utility functions for logging and visualization. Configuration files allow for easy experimentation with different parameters, while unit tests ensure robustness and code quality. This modular design facilitates further development, such as incorporating real-world data, exploring alternative RL algorithms, or extending the system to multi-agent scenarios.

Evaluation and Visualization:
The system tracks key performance metrics such as cumulative reward, user engagement rates, and diversity in recommendations. Visualization tools, such as custom dashboards or TensorBoard, help monitor training progress and provide insights into the agent's learning process.

Applications and Impact:

By integrating advanced RL methods into recommendation systems, this project addresses challenges in personalization and dynamic decision-making. The approach can be applied to various industries, from media and entertainment to e-commerce, enhancing user satisfaction and engagement through tailored content delivery.


