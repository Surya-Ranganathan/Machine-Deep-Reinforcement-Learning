import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, k=10, alpha=None):
        self.k = k  # Number of arms
        self.q_true = np.zeros(k)  # True action values (start equal)
        self.q_est = np.zeros(k)  # Estimated action values
        self.action_count = np.zeros(k)  # Count of each action selection
        self.alpha = alpha  # Step-size parameter (None for sample-average method)

    def step(self):
        # Random walk for each action value
        self.q_true += np.random.normal(0, 0.01, self.k)

    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1)

    def update_estimates(self, action, reward):
        self.action_count[action] += 1
        if self.alpha is None:
            # Sample-average update
            self.q_est[action] += (1 / self.action_count[action]) * (reward - self.q_est[action])
        else:
            # Constant step-size update
            self.q_est[action] += self.alpha * (reward - self.q_est[action])

    def select_action(self, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.k)  # Exploration
        else:
            return np.argmax(self.q_est)  # Exploitation

# Simulation parameters
steps = 10000
runs = 2000
epsilon = 0.1

# Store results
opt_action_counts_sample_avg = np.zeros(steps)
opt_action_counts_alpha = np.zeros(steps)

for run in range(runs):
    bandit_sample_avg = NonStationaryBandit()
    bandit_alpha = NonStationaryBandit(alpha=0.1)
    print(run)
    
    for step in range(steps):
        bandit_sample_avg.step()
        bandit_alpha.step()

        
        optimal_action = np.argmax(bandit_sample_avg.q_true)
        
        # Sample-average method
        action_sample_avg = bandit_sample_avg.select_action(epsilon)
        reward_sample_avg = bandit_sample_avg.get_reward(action_sample_avg)
        bandit_sample_avg.update_estimates(action_sample_avg, reward_sample_avg)
        
        # Constant step-size method
        action_alpha = bandit_alpha.select_action(epsilon)
        reward_alpha = bandit_alpha.get_reward(action_alpha)
        bandit_alpha.update_estimates(action_alpha, reward_alpha)
        
        # Record optimal action percentages
        if action_sample_avg == optimal_action:
            opt_action_counts_sample_avg[step] += 1
        if action_alpha == optimal_action:
            opt_action_counts_alpha[step] += 1

# Compute averages over runs
opt_action_counts_sample_avg /= runs
opt_action_counts_alpha /= runs

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(opt_action_counts_sample_avg * 100, label="Sample Average", color='red')
plt.plot(opt_action_counts_alpha * 100, label="Constant Step-size (α=0.1)", color='blue')
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Performance in Nonstationary 10-Armed Bandit")
plt.legend()
plt.show()
