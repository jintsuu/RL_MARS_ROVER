import random

import numpy as np
import matplotlib.pyplot as plt
from Mars_Rover_env import MarsRoverEnv
random.seed(42)

def alpha_schedule(e, max_e=1000, min_alpha=0.01):
    return max(min_alpha, (1 - e/max_e)*(1-min_alpha) + min_alpha)

def epsilon_schedule(e, max_e=2000, min_eps=0.1):
    return max(min_eps, (1 - e/max_e)*(1-min_eps) + min_eps)

#SARSA: record total reward per episode
def run_sarsa_rewards(env, n_eps, gamma=1.0):
    Q = np.zeros((env.nS, env.nA))
    rewards = np.zeros(n_eps)
    for ep in range(n_eps):
        α, ε = alpha_schedule(ep), epsilon_schedule(ep)
        s = env.reset()
        a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
        done = False
        ep_r = 0.0
        while not done:
            s2, r, done, _ = env.step(a)
            ep_r += r
            a2 = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s2])
            Q[s,a] += α*(r + gamma*Q[s2,a2] - Q[s,a])
            s, a = s2, a2
        rewards[ep] = ep_r
    return rewards

#Q-Learning
def run_q_rewards(env, n_eps, gamma=1.0):
    Q = np.zeros((env.nS, env.nA))
    rewards = np.zeros(n_eps)
    for ep in range(n_eps):
        α, ε = alpha_schedule(ep), epsilon_schedule(ep)
        s, done = env.reset(), False
        ep_r = 0.0
        while not done:
            a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
            s2, r, done, _ = env.step(a)
            ep_r += r
            Q[s,a] += α*(r + gamma*np.max(Q[s2]) - Q[s,a])
            s = s2
        rewards[ep] = ep_r
    return rewards

#First-Visit Monte Carlo
def run_mc_rewards(env, n_eps, gamma=1.0):
    Q = np.zeros((env.nS, env.nA))
    returns_count = np.zeros((env.nS, env.nA))
    rewards = np.zeros(n_eps)
    for ep in range(n_eps):
        ε = epsilon_schedule(ep)
        episode = []
        s, done = env.reset(), False
        while not done:
            a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
            s2, r, done, _ = env.step(a)
            episode.append((s,a,r))
            s = s2
        rewards[ep] = sum(r for (_,_,r) in episode)
        G = 0; visited = set()
        for (s_t,a_t,r_t) in reversed(episode):
            G = gamma*G + r_t
            if (s_t,a_t) not in visited:
                visited.add((s_t,a_t))
                returns_count[s_t,a_t] += 1
                Q[s_t,a_t] += (G - Q[s_t,a_t]) / returns_count[s_t,a_t]
    return rewards

#Double Q-Learning
def run_double_q_rewards(env, n_eps, gamma=1.0):
    Q1 = np.zeros((env.nS, env.nA))
    Q2 = np.zeros((env.nS, env.nA))
    rewards = np.zeros(n_eps)
    for ep in range(n_eps):
        α, ε = alpha_schedule(ep), epsilon_schedule(ep)
        s, done = env.reset(), False
        ep_r = 0.0
        while not done:
            if np.random.rand()<ε:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q1[s] + Q2[s])
            s2, r, done, _ = env.step(a)
            ep_r += r
            if np.random.rand()<0.5:
                a_max = np.argmax(Q1[s2])
                Q1[s,a] += α*(r + gamma*(not done)*Q2[s2,a_max] - Q1[s,a])
            else:
                a_max = np.argmax(Q2[s2])
                Q2[s,a] += α*(r + gamma*(not done)*Q1[s2,a_max] - Q2[s,a])
            s = s2
        rewards[ep] = ep_r
    return rewards

def moving_average(x, window=50):
    return np.array([np.mean(x[max(0,i-window+1):i+1]) for i in range(len(x))])

if __name__ == "__main__":
    n_episodes = 3000

    env = MarsRoverEnv(
        n_states=5,
        p_stay=1/3,
        p_backward=1/6,
        left_side_reward=1,
        right_side_reward=10
    )

    rew_sarsa = run_sarsa_rewards(env,    n_episodes)
    rew_q     = run_q_rewards(env,        n_episodes)
    rew_mc    = run_mc_rewards(env,       n_episodes)
    rew_dq    = run_double_q_rewards(env, n_episodes)

    eps = np.arange(n_episodes)
    ma_sarsa = moving_average(rew_sarsa)
    ma_q     = moving_average(rew_q)
    ma_mc    = moving_average(rew_mc)
    ma_dq    = moving_average(rew_dq)

    # plot
    plt.figure(figsize=(10,6))
    plt.plot(eps, ma_sarsa, label="SARSA")
    plt.plot(eps, ma_q,     label="Q-Learning")
    plt.plot(eps, ma_mc,    label="First-Visit MC")
    plt.plot(eps, ma_dq,    label="Double Q-Learning")
    plt.title("Mean Reward per Episode — Moving Avg window=50")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mean_reward.png")
    plt.show()
