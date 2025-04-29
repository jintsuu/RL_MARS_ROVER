import numpy as np
import matplotlib.pyplot as plt
from Mars_Rover_env import MarsRoverEnv

def run_td_prediction(n_states = 5, alpha = 0.1, snapshot_episodes=(0,1,10,100)):
    env = MarsRoverEnv(n_states=n_states)
    nS = env.nS

    V = np.zeros(nS)

    snapshots = {0: V[1:-1].copy()}
    max_ep = max(snapshot_episodes)

    for ep in range(1, max_ep+1):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            target = reward + (0.0 if done else V[next_state])
            V[state] += alpha * (target - V[state])
            state = next_state

        if ep in snapshot_episodes:
            snapshots[ep] = V[1:-1].copy()

    return snapshots

def plot_snapshots(snapshots, n_states = 5):
    states = np.arange(1, n_states+1)
    plt.figure(figsize=(8,6))
    for ep, vals in sorted(snapshots.items()):
        plt.plot(states, vals, marker='o', label=f'{ep} epi')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    snapshots = run_td_prediction()
    plot_snapshots(snapshots)