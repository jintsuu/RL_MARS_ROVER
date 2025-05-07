import numpy as np
import matplotlib.pyplot as plt
from Mars_Rover_env import MarsRoverEnv

# Parameter schedules for α and ε decay
def alpha_schedule(e, max_e=1000, min_alpha=0.01):
    return max(min_alpha, (1 - e/max_e)*(1-min_alpha) + min_alpha)

def epsilon_schedule(e, max_e=2000, min_eps=0.1):
    return max(min_eps, (1 - e/max_e)*(1-min_eps) + min_eps)

# SARSA
def run_sarsa(env, n_episodes, gamma=1.0):
    Q = np.zeros((env.nS, env.nA))
    values = np.zeros((n_episodes, env.nS-2))
    for ep in range(n_episodes):
        α = alpha_schedule(ep)
        ε = epsilon_schedule(ep)
        s = env.reset()
        a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
        done = False
        while not done:
            s2, r, done, _ = env.step(a)
            a2 = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s2])
            Q[s,a] += α*(r + gamma*Q[s2,a2] - Q[s,a])
            s, a = s2, a2
        values[ep] = [np.max(Q[s_]) for s_ in range(1, env.nS-1)]
    return values

# Q-Learning
def run_q_learning(env, n_episodes, gamma=1.0):
    Q = np.zeros((env.nS, env.nA))
    values = np.zeros((n_episodes, env.nS-2))
    for ep in range(n_episodes):
        α = alpha_schedule(ep)
        ε = epsilon_schedule(ep)
        s = env.reset()
        done = False
        while not done:
            a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
            s2, r, done, _ = env.step(a)
            Q[s,a] += α*(r + gamma*np.max(Q[s2]) - Q[s,a])
            s = s2
        values[ep] = [np.max(Q[s_]) for s_ in range(1, env.nS-1)]
    return values

# First-Visit Monte Carlo
def run_mc(env, n_episodes, gamma=1.0):
    Q = np.zeros((env.nS, env.nA))
    returns_count = np.zeros((env.nS, env.nA))
    values = np.zeros((n_episodes, env.nS-2))
    for ep in range(n_episodes):
        ε = epsilon_schedule(ep)
        episode = []
        s = env.reset()
        done = False
        while not done:
            a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
            s2, r, done, _ = env.step(a)
            episode.append((s,a,r))
            s = s2
        G = 0
        visited = set()
        for s_t,a_t,r_t in reversed(episode):
            G = gamma*G + r_t
            if (s_t,a_t) not in visited:
                visited.add((s_t,a_t))
                returns_count[s_t,a_t] += 1
                Q[s_t,a_t] += (G - Q[s_t,a_t]) / returns_count[s_t,a_t]
        values[ep] = [np.max(Q[s_]) for s_ in range(1, env.nS-1)]
    return values

# Double Q-Learning
def run_double_q_learning(env, n_episodes, gamma=1.0):
    Q1 = np.zeros((env.nS, env.nA))
    Q2 = np.zeros((env.nS, env.nA))
    values = np.zeros((n_episodes, env.nS-2))
    for ep in range(n_episodes):
        α = alpha_schedule(ep)
        ε = epsilon_schedule(ep)
        s = env.reset()
        done = False
        while not done:
            if np.random.rand() < ε:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q1[s] + Q2[s])
            s2, r, done, _ = env.step(a)
            if np.random.rand() < 0.5:
                a_max = np.argmax(Q1[s2])
                target = r + gamma * Q2[s2, a_max]
                Q1[s, a] += α * (target - Q1[s, a])
            else:
                a_max = np.argmax(Q2[s2])
                target = r + gamma * Q1[s2, a_max]
                Q2[s, a] += α * (target - Q2[s, a])
            s = s2
        values[ep] = [
            np.max((Q1[s_] + Q2[s_]) / 2.0) for s_ in range(1, env.nS-1)
        ]
    return values

def moving_average(x, window=100):
    return np.array([np.mean(x[max(0,i-window+1):i+1]) for i in range(len(x))])

if __name__ == "__main__":
    n_eps = 3000
    eps_range = np.arange(1, n_eps+1)

    env = MarsRoverEnv(
        n_states=5,
        p_stay=1/3,
        p_backward=1/6,
        left_side_reward=1,
        right_side_reward=10
    )

    vals_sarsa = run_sarsa(env, n_eps)
    vals_q     = run_q_learning(env, n_eps)
    vals_mc    = run_mc(env, n_eps)
    vals_dq    = run_double_q_learning(env, n_eps)

    w = 50
    sm_sarsa = np.vstack([moving_average(vals_sarsa[:,i], w) for i in range(vals_sarsa.shape[1])]).T
    sm_q     = np.vstack([moving_average(vals_q[:,i],     w) for i in range(vals_q.shape[1])]).T
    sm_mc    = np.vstack([moving_average(vals_mc[:,i],    w) for i in range(vals_mc.shape[1])]).T
    sm_dq    = np.vstack([moving_average(vals_dq[:,i],    w) for i in range(vals_dq.shape[1])]).T

    # Plot
    fig, axs = plt.subplots(4,1, figsize=(10,20), sharex=True)
    methods = [
        (sm_sarsa, "SARSA"),
        (sm_q,     "Q-Learning"),
        (sm_mc,    "First-Visit MC"),
        (sm_dq,    "Double Q-Learning"),
    ]
    for ax, (data, name) in zip(axs, methods):
        for idx in range(data.shape[1]):
            ax.plot(eps_range, data[:,idx], label=f'state {idx+1}')
        ax.set_title(f'{name}: V(s) (Moving Avg window={w})')
        ax.set_ylabel('Estimated V')
        ax.legend()
        ax.grid(True)

    axs[-1].set_xlabel('Episode')
    plt.tight_layout()
    plt.savefig('compare_methods.png')
    plt.show()

