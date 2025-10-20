Morgan: Reinforcement Learning Agent for Minecraft

Morgan is a reinforcement learning agent built in Microsoft Malmo that learns to gather ingredients, craft items, cook food, and present the best possible gift.
The project began with a tabular Q-learning baseline and expanded into a Deep Q-Learning (DQN) prototype using PyTorch.

Overview

The goal of this project is to train an AI agent to:

Collect items (e.g., pumpkin, egg, sugar)

Craft recipes like pumpkin pie and mushroom stew

Cook raw ingredients into cooked food

Present the final crafted or cooked item to maximize reward

Morgan learns by interacting with the Minecraft world and updating its policy through Q-learning and DQN techniques.

Setup

Create environment

conda create -n myenv python=3.7
conda activate myenv


Install dependencies

pip install torch==1.10.2 numpy==1.21.6 matplotlib==3.5.3 gym==0.21.0


Install Microsoft Malmo

Download Malmo 0.37.0 for Python 3.7.

Copy MalmoPython.pyd into your environment’s Lib/site-packages.

Add Malmo’s /bin directory to your PATH.

Run the agent

python Python_Examples/final_project_runner.py

Key Features

Tabular Q-Learning baseline
Discrete state-action mapping using sorted inventory tuples.

Deep Q-Learning (DQN) prototype
Neural Q-network approximates Q(s, a) for generalization to unseen states.

Crafting and Cooking System
Automatically detects valid recipes and checks proximity to crafting tables or furnaces.

Pathfinding
A* search (a_star.py) enables efficient movement to targets.

Logging
Tracks per-episode rewards, ε values, and loss for performance evaluation.

Evaluating Progress

After training, you can visualize performance:

import numpy as np, matplotlib.pyplot as plt
rews = np.load("episode_rewards.npy")
plt.plot(rews)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Morgan Agent Learning Curve")
plt.show()


This shows how rewards improve over time as the agent learns better strategies.

Future Improvements

Integrate Double / Dueling DQN architectures

Implement Prioritized Experience Replay

Extend to Recurrent DQN (LSTM) for partial observability

Add action masking for invalid moves

Improve data visualization with TensorBoard or W&B
