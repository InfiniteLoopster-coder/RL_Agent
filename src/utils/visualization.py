# src/utils/visualization.py
import matplotlib.pyplot as plt

def plot_rewards(reward_history, title="Training Rewards", xlabel="Episode", ylabel="Reward"):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Example usage (uncomment to test locally):
# if __name__ == "__main__":
#     sample_rewards = [np.random.random() for _ in range(50)]
#     plot_rewards(sample_rewards)
