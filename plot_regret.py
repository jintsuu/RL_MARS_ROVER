import numpy as np
import matplotlib.pyplot as plt
from Mars_Rover_env import MarsRoverEnv

env_args = dict(
    n_states=5,
    p_stay=1/3,
    p_backward=1/6,
    left_side_reward=1,
    right_side_reward=10
)
base_env = MarsRoverEnv(**env_args)
nS, nA = base_env.nS, base_env.nA

def compute_optimal_Q(env, gamma=1.0, theta=1e-8):
    Q_star = np.zeros((env.nS, env.nA))
    while True:
        delta = 0.0
        for s in range(1, env.nS-1):
            for a in range(env.nA):
                old = Q_star[s, a]
                new = 0.0
                for (p, s2, r, done) in env.P[s][a]:
                    new += p * (r + gamma * (0 if done else np.max(Q_star[s2])))
                Q_star[s, a] = new
                delta = max(delta, abs(old-new))
        if delta < theta:
            break
    return Q_star

Q_star = compute_optimal_Q(base_env)

def run_sarsa_Q(env, n_eps):
    Q = np.zeros((nS,nA)); regrets = []
    for ep in range(n_eps):
        α = max(0.01, (1-ep/1000)*(1-0.01)+0.01)
        ε = max(0.1,  (1-ep/2000)*(1-0.1) +0.1)
        s = env.reset()
        a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
        done=False
        while not done:
            s2,r,done,_ = env.step(a)
            a2 = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s2])
            Q[s,a] += α*(r + Q[s2,a2] - Q[s,a])
            s,a = s2,a2
        diff = np.abs(Q[1:-1] - Q_star[1:-1])
        regrets.append(diff.mean())
    return np.array(regrets)

def run_q_Q(env, n_eps):
    Q = np.zeros((nS,nA)); regrets=[]
    for ep in range(n_eps):
        α = max(0.01, (1-ep/1000)*(1-0.01)+0.01)
        ε = max(0.1,  (1-ep/2000)*(1-0.1) +0.1)
        s,done = env.reset(), False
        while not done:
            a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
            s2,r,done,_ = env.step(a)
            Q[s,a] += α*(r + np.max(Q[s2]) - Q[s,a])
            s = s2
        diff = np.abs(Q[1:-1] - Q_star[1:-1])
        regrets.append(diff.mean())
    return np.array(regrets)

def run_mc_Q(env, n_eps, gamma=1.0):
    Q = np.zeros((nS,nA))
    returns_count = np.zeros((nS,nA))
    regrets=[]
    for ep in range(n_eps):
        ε = max(0.1,  (1-ep/2000)*(1-0.1) +0.1)
        episode=[]; s,done = env.reset(),False
        while not done:
            a = env.action_space.sample() if np.random.rand()<ε else np.argmax(Q[s])
            s2,r,done,_ = env.step(a)
            episode.append((s,a,r))
            s = s2
        G=0; seen=set()
        for (s_t,a_t,r_t) in reversed(episode):
            G = gamma*G + r_t
            if (s_t,a_t) not in seen:
                seen.add((s_t,a_t))
                returns_count[s_t,a_t]+=1
                Q[s_t,a_t] += (G - Q[s_t,a_t]) / returns_count[s_t,a_t]
        diff = np.abs(Q[1:-1] - Q_star[1:-1])
        regrets.append(diff.mean())
    return np.array(regrets)

def run_double_q_Q(env, n_eps, gamma=1.0):
    Q1 = np.zeros((nS,nA))
    Q2 = np.zeros((nS,nA))
    regrets=[]
    for ep in range(n_eps):
        α = max(0.01, (1-ep/1000)*(1-0.01)+0.01)
        ε = max(0.1,  (1-ep/2000)*(1-0.1) +0.1)
        s,done = env.reset(),False
        while not done:
            if np.random.rand()<ε:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q1[s] + Q2[s])
            s2,r,done,_ = env.step(a)
            if np.random.rand()<0.5:
                a_max = np.argmax(Q1[s2])
                Q1[s,a] += α*(r + (0 if done else Q2[s2,a_max]) - Q1[s,a])
            else:
                a_max = np.argmax(Q2[s2])
                Q2[s,a] += α*(r + (0 if done else Q1[s2,a_max]) - Q2[s,a])
            s = s2
        Q = (Q1+Q2)/2.0
        diff = np.abs(Q[1:-1] - Q_star[1:-1])
        regrets.append(diff.mean())
    return np.array(regrets)

n_episodes = 3000
runs = {
    'SARSA':   run_sarsa_Q,
    'Q-Learning': run_q_Q,
    'First-Visit MC': run_mc_Q,
    'Double Q-Learning': run_double_q_Q
}

window = 50
plt.figure(figsize=(10,6))
for name, runner in runs.items():
    env = MarsRoverEnv(**env_args)
    reg = runner(env, n_episodes)
    ma = np.array([reg[max(0,i-window+1):i+1].mean() for i in range(n_episodes)])
    plt.plot(ma, label=name)

plt.title('Regret = meanₛₐ |Qₜ(s,a)−Q* (s,a)| — Moving Avg window=50')
plt.xlabel('Episode')
plt.ylabel('Mean absolute Q-error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_regret.png')
plt.show()
