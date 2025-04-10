# Introduction2RL_2024


Additional material for the lecture and exercises held at Paris Lodron University Salzburg in 2024 by me (Simon Hirlaender).


This is our little Mars rover from the lecture:
<img width="1012" alt="image" src="https://github.com/MathPhysSim/Introduction2RL_2024/assets/22523245/419f30be-12f0-4445-a077-56b0c8f03eda">



# Mars Rover Environment with Gymnasium

The Mars Rover Environment is a reinforcement learning scenario implemented with Gymnasium. This environment simulates a rover navigating across a linear Martian landscape, aiming to reach a designated goal. It's designed to provide a simple yet challenging task for reinforcement learning algorithms, focusing on probabilistic state transitions and reward optimization.

## Environment Overview

In this simulated Martian landscape, the rover faces a series of states it can navigate through by moving left or right. The environment is discrete, with terminal states at both ends of the linear space. The rover's goal is to reach the most rewarding terminal state, overcoming the uncertainty introduced by probabilistic movement outcomes.

### States

- The environment is composed of `n_states + 2` states, where `n_states` is the number of non-terminal states. The two additional states represent the terminal points at either end of the rover's path.
- The rover's position within these states determines its progress and influences the rewards it accumulates.

### Actions

The rover can perform two actions, each intended to move it one step in the desired direction:
- **Left (0)**: Move one state to the left.
- **Right (1)**: Move one state to the right.

The actual outcome of an action is subject to probabilities, adding an element of unpredictability to the rover's movement.

### Transition Probabilities

The movement outcomes are influenced by the following probabilities:
- **p_stay**: The probability that the rover remains in its current state, despite taking an action.
- **p_backward**: The probability that the rover moves in the opposite direction of the intended action.
- The probability of moving forward, as intended, is `1 - p_stay - p_backward`, ensuring all probabilities sum to 1.

### Rewards

- **left_side_reward**: Reward received upon reaching the left terminal state.
- **right_side_reward**: Higher reward for reaching the right terminal state, incentivizing the rover to navigate towards this goal.

### Terminal States

Reaching a terminal state concludes the current episode. These states represent the rover's successful navigation to an endpoint of its journey, with rewards allocated based on the terminal state reached.

## Integration with Gymnasium

This environment is compatible with the Gymnasium library, facilitating its use in reinforcement learning projects. Gymnasium provides a standardized API for interacting with the environment, including initiating episodes, taking actions, and receiving feedback in the form of state observations, rewards, and termination signals.

## Customization

The Mars Rover Environment supports several customization options, allowing users to adjust the number of states (`n_states`), the probabilities of action outcomes (`p_stay`, `p_backward`), and the rewards for reaching terminal states (`left_side_reward`, `right_side_reward`). This flexibility makes it suitable for various experimentation needs, from introductory reinforcement learning tasks to more complex strategic explorations.

## Usage Note

To utilize this environment, ensure you have the Gymnasium library installed in your Python environment (`pip install gymnasium`). The environment inherits from Gymnasium's `DiscreteEnv`, using its mechanisms for discrete state and action spaces, probabilistic transitions, and reward definitions.
