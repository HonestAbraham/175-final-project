import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
import math

# Remove all signal handling code
# Global flag for interruption
# should_stop = False
# 
# def signal_handler(signum, frame):
#     """Handle interruption signal"""
#     global should_stop
#     should_stop = True
#     print("\nInterruption detected! Analyzing current state...")
# 
# # Register the signal handler
# signal.signal(signal.SIGINT, signal_handler)

def check_stop_learning(episode: int) -> bool:
    """Check if we should stop learning based on episode number"""
    if episode % 10 == 0:
        response = input(f"\nEpisode {episode} completed. Would you like to stop? (y/n): ").lower()
        return response in ['y', 'yes']
    return False

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def state_to_tensor(state: Tuple) -> torch.Tensor:
    """Convert state tuple to tensor for DQN"""
    # Create a one-hot encoding of the state
    state_dict = dict(state)
    tensor = torch.zeros(len(items))
    for item, count in state_dict.items():
        if item in items:
            tensor[items.index(item)] = count
    return tensor

def analyze_learning_state(q_table: Dict = None, dqn_model: DQN = None, 
                          state_mapping: Dict = None, is_dqn: bool = False):
    """Analyze current state of learning for both Q-learning and DQN"""
    print("\n=== Learning State Analysis ===")
    
    if is_dqn and dqn_model is not None:
        print("DQN Model Analysis:")
        # Convert all possible states to tensors and get predictions
        all_states = []
        all_rewards = []
        
        # Sample some states to analyze
        for state in state_mapping.keys():
            state_tensor = state_to_tensor(state)
            with torch.no_grad():
                q_values = dqn_model(state_tensor)
                best_action_idx = torch.argmax(q_values).item()
                best_reward = q_values[best_action_idx].item()
                
                all_states.append((state, best_action_idx, best_reward))
        
        # Sort by reward
        all_states.sort(key=lambda x: x[2], reverse=True)
        
        print("\nTop 5 DQN Strategies:")
        for i, (state, action_idx, reward) in enumerate(all_states[:5]):
            print(f"\nStrategy {i+1}:")
            print(f"Current Items: {dict(state)}")
            print(f"Recommended Action: {state_mapping[action_idx]}")
            print(f"Expected Reward: {reward:.2f}")
            
            action = state_mapping[action_idx]
            if action in food_recipes:
                print(f"Creates: {action} using {food_recipes[action]}")
            elif action in cooking_recipes:
                print(f"Creates: {action} using {cooking_recipes[action]}")
    
    if not is_dqn and q_table is not None:
        print("Tabular Q-Learning Analysis:")
        best_states = []
        best_actions = []
        best_rewards = []
        state_visits = {}
        
        for state, actions in q_table.items():
            if not actions:
                continue
            state_visits[state] = sum(1 for action in actions.values() if action != 0)
            best_action = max(actions.items(), key=lambda x: x[1])
            best_states.append(state)
            best_actions.append(best_action[0])
            best_rewards.append(best_action[1])
        
        sorted_indices = sorted(range(len(best_rewards)), 
                              key=lambda i: best_rewards[i], reverse=True)
        
        print("\nTop 5 Q-Learning Strategies:")
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            state = best_states[idx]
            action = best_actions[idx]
            reward = best_rewards[idx]
            visits = state_visits[state]
            
            print(f"\nStrategy {i+1}:")
            print(f"Current Items: {dict(state)}")
            print(f"Recommended Action: {action}")
            print(f"Expected Reward: {reward:.2f}")
            print(f"State Visit Count: {visits}")
            
            if action in food_recipes:
                print(f"Creates: {action} using {food_recipes[action]}")
            elif action in cooking_recipes:
                print(f"Creates: {action} using {cooking_recipes[action]}")
        
        print(f"\nQ-Table Statistics:")
        print(f"Total States: {len(q_table)}")
        print(f"Total State-Action Pairs: {sum(len(actions) for actions in q_table.values())}")

def save_learning_state(q_table: Dict = None, dqn_model: DQN = None, 
                       state_mapping: Dict = None, is_dqn: bool = False,
                       filename: str = "learning_state.txt"):
    """Save current learning state to file"""
    with open(filename, 'w') as f:
        if is_dqn and dqn_model is not None:
            f.write("=== DQN Model State ===\n")
            # Save model architecture and parameters
            f.write(str(dqn_model))
            f.write("\n\nState-Action Mapping:\n")
            for idx, action in state_mapping.items():
                f.write(f"{idx}: {action}\n")
        elif not is_dqn and q_table is not None:
            f.write("=== Q-Table State ===\n")
            for state, actions in q_table.items():
                f.write(f"State: {state}\n")
                for action, value in actions.items():
                    f.write(f"  Action: {action.ljust(20)} Q-value: {value:.2f}\n")
                f.write("\n")
    
    # Analyze and display current state
    analyze_learning_state(q_table, dqn_model, state_mapping, is_dqn)

items = [
    'beef',
    'porkchop',
    'fish',
    'rabbit',
    'coal',
    'pumpkin',
    'planks',
    'planks',
    'wheat',
    'egg',
    'sugar',
    'apple',
    'carrot',
    'potato'
]


food_recipes = {
    'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
    'pumpkin_seeds': ['pumpkin'],
    'bowl': ['planks', 'planks'],
    'mushroom_stew': ['bowl', 'red_mushroom'],
    'bread': ['wheat', 'wheat', 'wheat'],
    'cake': ['wheat', 'wheat', 'wheat', 'sugar', 'egg', 'milk'],
    'cookie': ['wheat', 'cocoa_beans'],
    'apple_pie': ['apple', 'sugar', 'wheat']
}



cooking_recipes = {
    'cooked_beef': ['coal', 'beef'],
    'cooked_porkchop': ['coal', 'porkchop'],
    'cooked_rabbit': ['coal', 'rabbit'],
    'cooked_fish': ['coal', 'fish'],
    'baked_potato': ['coal', 'potato'],
    'cooked_carrot': ['coal', 'carrot']
}


rewards_map = {
    # Raw food items
    'beef': 3,
    'porkchop': 3,
    'rabbit': 3,
    'fish': 2,
    'coal': 0,
    'pumpkin': 0,
    'wheat': 1,
    'egg': 1,
    'sugar': 0,
    'apple': 2,
    'carrot': 2,
    'potato': 2,
    'planks': 0,
    
    # Cooked food items
    'cooked_beef': 8,
    'cooked_porkchop': 8,
    'cooked_rabbit': 5,
    'cooked_fish': 5,
    'baked_potato': 5,
    'cooked_carrot': 3,
    
    # Crafted food items
    'pumpkin_pie': 8,
    'pumpkin_seeds': 1,
    'bread': 5,
    'cookie': 2,
    'apple_pie': 6,
    'bowl': 1,
    'mushroom_stew': 6
}


def get_curr_state(items):
    filtered = [(k, v) for k, v in items if v > 0]
    return tuple(sorted(filtered))


def choose_action(curr_state, possible_actions, eps, q_table):
    save_q_table_to_file(q_table)
    if curr_state not in q_table:
        q_table[curr_state] = {action: 0 for action in possible_actions}

    for action in possible_actions:
        if action not in q_table[curr_state]:
            q_table[curr_state][action] = 0

    if random.random() < eps:
        return random.choice(possible_actions)
    else:
        q_values = q_table[curr_state]
        max_q = max(q_values[a] for a in possible_actions)
        best_actions = [a for a in possible_actions if q_values[a] == max_q]
        chosen_action = random.choice(best_actions) 
        
        return chosen_action


def analyze_best_path(q_table):
    """Analyze the Q-table to find the best path to maximum reward for tabular Q-learning."""
    best_states = []
    best_actions = []
    best_rewards = []
    state_visits = {}  # Track how many times each state was visited
    
    # Find states with highest Q-values and count visits
    for state, actions in q_table.items():
        if not actions:  # Skip empty action sets
            continue
            
        # Count total visits to this state
        state_visits[state] = sum(1 for action in actions.values() if action != 0)
        
        best_action = max(actions.items(), key=lambda x: x[1])
        best_states.append(state)
        best_actions.append(best_action[0])
        best_rewards.append(best_action[1])
    
    # Sort by reward value
    sorted_indices = sorted(range(len(best_rewards)), key=lambda i: best_rewards[i], reverse=True)
    
    print("\n=== Tabular Q-Learning Analysis ===")
    print("Top 5 recommended strategies:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        state = best_states[idx]
        action = best_actions[idx]
        reward = best_rewards[idx]
        visits = state_visits[state]
        
        print(f"\nStrategy {i+1}:")
        print(f"Current Items: {dict(state)}")
        print(f"Recommended Action: {action}")
        print(f"Expected Reward: {reward:.2f}")
        print(f"State Visit Count: {visits}")
        
        # Show what the action produces
        if action in food_recipes:
            print(f"Creates: {action} using {food_recipes[action]}")
        elif action in cooking_recipes:
            print(f"Creates: {action} using {cooking_recipes[action]}")
    
    # Print some statistics about the Q-table
    total_states = len(q_table)
    total_actions = sum(len(actions) for actions in q_table.values())
    print(f"\nQ-Table Statistics:")
    print(f"Total States: {total_states}")
    print(f"Total State-Action Pairs: {total_actions}")
    print(f"Average Actions per State: {total_actions/total_states:.2f}")

def save_q_table_to_file(q_table, filename="q_table_output.txt"):
    with open(filename, 'w') as f:
        for state, actions in q_table.items():
            f.write(f"State: {state}\n")
            for action, value in actions.items():
                f.write(f"  Action: {action.ljust(20)} Q-value: {value:.2f}\n")
            f.write("\n")
    
    # After saving, analyze and display the best paths
    analyze_best_path(q_table)

def is_solution(reward):
    return reward >= 11