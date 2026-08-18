# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate one complete episode
    episode = generate_episode(epsilon)

    # Store reward obtained in this episode
    total_reward = sum(step[2] for step in episode)
    episode_rewards.append(total_reward)

    # Return from the end of the episode
    G = 0

    # Visit every state-action pair only once
    visited = set()

    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        if (state, action) not in visited:

            visited.add((state, action))

            # Incremental Monte Carlo update
            Q[state, action] = Q[state, action] + alpha * (
                G - Q[state, action]
            )

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name:Jothi Ganesh P         ")
    print("Register Number: 212224240065    ")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(optimal_policy)


success_rate = np.mean(episode_rewards[-1000:])

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title("Monte Carlo Control Learning Curve")

plt.grid(True)

plt.show()


env.close()

```



## Output

Final Q-table:



<img width="597" height="387" alt="image" src="https://github.com/user-attachments/assets/07607e10-08a0-4177-89f9-5f97dd3e8cc9" />



Estimated State-Value Function:


<img width="705" height="183" alt="image" src="https://github.com/user-attachments/assets/9cdfde72-6273-4697-95e2-7dd38b3fe10b" />








Learned Policy:

<img width="432" height="125" alt="image" src="https://github.com/user-attachments/assets/90120084-c79d-41d4-ae7f-c90afaf19cd2" />




Average reward over last 1000 episodes: 



<img width="500" height="46" alt="image" src="https://github.com/user-attachments/assets/656ad011-0ed8-4baa-b489-8ab131e51483" />


<img width="943" height="618" alt="image" src="https://github.com/user-attachments/assets/7ea251ff-66db-4664-86bc-1cab9c38b9c9" />


## Result

The On-Policy Monte Carlo Control algorithm was successfully implemented using Gymnasium's FrozenLake-v1 environment. The agent learned an improved policy using Monte Carlo returns and an epsilon-greedy strategy.



## Inference

The agent initially performs more exploration because the epsilon value starts at 1.0. As the number of training episodes increases, epsilon decreases toward the minimum value of 0.05.

The Monte Carlo algorithm calculates the return from complete episodes and uses these returns to update the Q-values.

The learned Q-table represents the estimated value of taking each action in every state. The state-value function is obtained by selecting the maximum Q-value for each state.

The epsilon-greedy policy allows the agent to explore different actions initially and gradually exploit the actions that provide higher estimated returns.

The learning curve shows the change in the average reward as the number of training episodes increases.





---

