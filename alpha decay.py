import numpy as np
import matplotlib.pyplot as plt
from Mars_Rover_env import MarsRoverEnv

def alpha_schedule(t, t_max=250, a_start=0.5, a_end=0.01):
    decay = (a_start - a_end) / (t_max - 1)
    return max(a_end, a_start - decay * (t - 1))

def run_td_decay(env, t_max):
    nS = env.nS
    V = np.zeros(nS)
    V_hist = np.zeros((5, t_max + 1))
    V_hist[:, 0] = V[1:-1]

    for t in range(1, t_max + 1):
        alpha = alpha_schedule(t, t_max)
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            target = reward + (0 if done else V[next_state])
            V[state] += alpha * (target - V[state])
            state = next_state
        V_hist[:, t] = V[1:-1]

    return V_hist

def run_mc_decay(env, t_max):
    nS = env.nS
    V = np.zeros(nS)
    V_hist = np.zeros((5, t_max + 1))
    V_hist[:, 0] = V[1:-1]

    for t in range(1, t_max + 1):
        alpha = alpha_schedule(t, t_max)
        state = env.reset()
        done = False
        episode_states = []
        rewards = []
        while not done:
            episode_states.append(state)
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

        G = sum(rewards)
        first_visits = set()
        for idx, s in enumerate(episode_states):
            if s not in first_visits and 1 <= s <= nS - 2:
                first_visits.add(s)
                V[s] += alpha * (G - V[s])
        V_hist[:, t] = V[1:-1]

    return V_hist

def true_values_linear_system(n_states):
    env = MarsRoverEnv(n_states=n_states)
    P = np.zeros((n_states, n_states))
    R = np.zeros(n_states)
    for s in range(1, n_states+1):
        for a in (0,1):
            for prob, ns, rew, done in env.P[s][a]:
                weight = 0.5 * prob
                if 1 <= ns <= n_states:
                    P[s-1, ns-1] += weight
                else:
                    R[s-1] += weight * rew
    V_int = np.linalg.solve(np.eye(n_states) - P, R)
    V = np.zeros(n_states+2)
    V[1:-1] = V_int
    V[0], V[-1] = 1.0, 10.0
    return V

if __name__ == '__main__':
    runs = 100
    t_max = 250
    n_states = 5

    td_sum = np.zeros((5, t_max + 1))
    mc_sum = np.zeros((5, t_max + 1))

    for _ in range(runs):
        env_td = MarsRoverEnv(n_states=n_states)
        env_mc = MarsRoverEnv(n_states=n_states)
        td_sum += run_td_decay(env_td, t_max)
        mc_sum += run_mc_decay(env_mc, t_max)

    td_avg = td_sum / runs
    mc_avg = mc_sum / runs

    V_true = true_values_linear_system(n_states)
    true_nonterm = V_true[1:-1]

    episodes = np.arange(0, t_max + 1)

    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    for i, ax in enumerate(axes, start=1):
        ax.plot(episodes, td_avg[i-1], label='TD(0) avg')
        ax.plot(episodes, mc_avg[i-1], '--', label='MC avg')
        ax.axhline(true_nonterm[i-1], color='black', linestyle=':', label='True V(s)' if i==1 else None)
        ax.set_ylabel(f'$V(s={i})$')
        ax.grid(True)
        if i == 1:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Episode')
    plt.suptitle(r'Averaged over 100 runs, Decaying $\alpha$: 0.5 $\to$ 0.01')
    plt.tight_layout()
    plt.savefig('alpha_decay.png')
    plt.show()
