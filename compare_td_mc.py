import numpy as np
import matplotlib.pyplot as plt
from Mars_Rover_env import MarsRoverEnv

def true_values_analytic(n_states):
    N = n_states + 1
    V = np.zeros(n_states+2)
    for s in range(1, n_states+1):
        V[s] = 1 + 9 * (s / N)
    V[0], V[-1] = 1.0, 10.0
    return V

def true_values_linear_system(n_states):
    env = MarsRoverEnv(n_states=n_states)
    nS = env.nS
    P = np.zeros((n_states, n_states))
    R = np.zeros(n_states)
    for s in range(1, n_states+1):
        for a in (0,1):
            for prob, ns, rew, done in env.P[s][a]:
                weight = 0.5 * prob
                if 1 <= ns <= n_states:
                    P[s-1, ns-1] += weight
                else:
                    # terminal next-state
                    R[s-1] += weight * rew
    # Solve (I - P)V_int = R
    V_int = np.linalg.solve(np.eye(n_states) - P, R)
    V = np.zeros(n_states+2)
    V[1:-1] = V_int
    V[0], V[-1] = 1.0, 10.0
    return V

def run_td(env, alpha, max_episodes, V0, V_ref):
    V = np.full_like(V_ref, V0)
    V[0], V[-1] = V_ref[0], V_ref[-1]
    errs = np.zeros(max_episodes)
    for ep in range(max_episodes):
        s = env.reset(); done=False
        while not done:
            a = env.action_space.sample()
            ns, r, done, _ = env.step(a)
            target = r + (0.0 if done else V[ns])
            V[s] += alpha * (target - V[s])
            s = ns
        errs[ep] = np.sqrt(np.mean((V[1:-1] - V_ref[1:-1])**2))
    return errs

def run_mc(env, alpha, max_episodes, V0, V_ref):
    V = np.full_like(V_ref, V0)
    V[0], V[-1] = V_ref[0], V_ref[-1]
    errs = np.zeros(max_episodes)
    for ep in range(max_episodes):
        s = env.reset(); done=False
        states, rewards = [], []
        while not done:
            states.append(s)
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            rewards.append(r)
        G = sum(rewards)
        firsts = set()
        for i, s0 in enumerate(states):
            if s0 not in firsts and 1 <= s0 <= env.nS-2:
                firsts.add(s0)
                V[s0] += alpha * (G - V[s0])
        errs[ep] = np.sqrt(np.mean((V[1:-1] - V_ref[1:-1])**2))
    return errs

if __name__ == '__main__':
    n_states     = 5
    max_episodes = 100
    runs         = 100
    init_V       = 5.5
    alphas       = [0.005, 0.01, 0.02, 0.03]

    V_analytic = true_values_analytic(n_states)
    V_linear   = true_values_linear_system(n_states)

    td_errs_A, mc_errs_A = {α: np.zeros(max_episodes) for α in alphas}, {α: np.zeros(max_episodes) for α in alphas}
    td_errs_L, mc_errs_L = {α: np.zeros(max_episodes) for α in alphas}, {α: np.zeros(max_episodes) for α in alphas}

    for α in alphas:
        for _ in range(runs):
            e_td_A = MarsRoverEnv(n_states=n_states)
            e_mc_A = MarsRoverEnv(n_states=n_states)
            td_errs_A[α] += run_td(e_td_A, α, max_episodes, init_V, V_analytic)
            mc_errs_A[α] += run_mc(e_mc_A, α, max_episodes, init_V, V_analytic)

            e_td_L = MarsRoverEnv(n_states=n_states)
            e_mc_L = MarsRoverEnv(n_states=n_states)
            td_errs_L[α] += run_td(e_td_L, α, max_episodes, init_V, V_linear)
            mc_errs_L[α] += run_mc(e_mc_L, α, max_episodes, init_V, V_linear)

        td_errs_A[α] /= runs;  mc_errs_A[α] /= runs
        td_errs_L[α] /= runs;  mc_errs_L[α] /= runs

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    episodes = np.arange(1, max_episodes+1)

    #analytic reference
    for α in alphas:
        axes[0,0].plot(episodes, td_errs_A[α], label=f'α={α:.3f}')
        axes[0,1].plot(episodes, mc_errs_A[α], label=f'α={α:.3f}')
    axes[0,0].set_title('TD(0) vs Analytic')
    axes[0,1].set_title('MC vs Analytic')

    #linear‐system reference
    for α in alphas:
        axes[1,0].plot(episodes, td_errs_L[α], label=f'α={α:.3f}')
        axes[1,1].plot(episodes, mc_errs_L[α], label=f'α={α:.3f}')
    axes[1,0].set_title('TD(0) vs Linear‐System')
    axes[1,1].set_title('MC vs Linear‐System')

    for ax in axes.flat:
        ax.set_xlabel('Episode')
        ax.set_ylabel('RMS Error')
        ax.grid(True)
        ax.legend()

    plt.suptitle('RMS Error over 100 runs (init V=5.5), non‐terminals 1…5')
    plt.tight_layout()
    plt.savefig('compare_td_mc.png')
    plt.show()
