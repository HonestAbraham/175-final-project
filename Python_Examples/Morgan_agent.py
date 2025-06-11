from __future__ import division

import sys
import time
import json
import math
import hunger_learner_helper as submission
from collections import defaultdict, deque
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from collections import deque
import numpy as np  


from dqn_architecture import SimpleDQN, ReplayBuffer

BATCH_SIZE = 32
TARGET_UPDATE = 10   # how often (in episodes) to copy policy_net → target_net
MAX_STEPS_PER_EPISODE = 1000

items=submission.items
inventory_limit = 3
food_recipes = submission.food_recipes
rewards_map = submission.rewards_map
cooking_recipe = submission.cooking_recipes

class Morgan(object):
    def __init__(self, alpha=0.3, gamma=1, n=1, use_dqn=False):
        """Constructing an RL agent.

        Args
            alpha:  <float>  learning rate      (default = 0.3)
            gamma:  <float>  value decay rate   (default = 1)
            n:      <int>    number of back steps to update (default = 1)
        """
        self.use_dqn = use_dqn
        self.episode_rewards = []  # will store total reward per episode
        self.loss_history    = []  # will store loss value per optimization step

        # Initialize pathfinder
        from a_star import MorganPathfinder
        self.pathfinder = MorganPathfinder()

        if not use_dqn:
            # ───── Tabular Q‐learning branch (unchanged) ─────
            self.epsilon = 0.2
            self.q_table = {}
            self.n, self.alpha, self.gamma = n, alpha, gamma
            self.inventory = defaultdict(lambda: 0)
            self.num_items_in_inv = 0
            return

        # ───── NEW DQN SETUP ─────
        self.inventory = defaultdict(lambda: 0)
        self.num_items_in_inv = 0

        # 1) Build a combined list of raw + cooked item‐names
        # ── UPDATED ──
        raw_items    = submission.items[:]                           
        cooked_items = list(submission.cooking_recipes.keys())       # ←==== use keys(), not values()
        # Union them (avoid duplicates):
        self.item_list      = raw_items + [ci for ci in cooked_items if ci not in raw_items]
        self.num_item_types = len(self.item_list)
        # ──────────────────────────────────
           # e.g. now = 6 or 7

        self.inventory_limit = 3
        self.gamma           = 0.99
        self.eps_start       = 1.0
        self.eps_end         = 0.1
        self.eps_decay       = 20000
        self.epsilon         = self.eps_start

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build a fixed list of all possible actions (fetch, craft, cook, present_gift):
        self.all_actions = self._build_all_actions_list()
        self.num_actions = len(self.all_actions)

        # Initialize policy_net and target_net
        hidden_dim = 128
        self.policy_net = SimpleDQN(self.num_item_types, hidden_dim, self.num_actions).to(self.device)
        self.target_net = SimpleDQN(self.num_item_types, hidden_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Replay buffer + optimizer + loss
        self.memory    = ReplayBuffer(capacity=10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        self.episode_counter = 0    # ←–– initialize so you can safely do self.episode_counter+1 in run()


    def _build_all_actions_list(self):
        actions = []
        # (1) Every "fetch" action for raw items:
        for it in submission.items:
            actions.append(it)
        # (2) Every "craft" action "c_<food_name>":
        for k in submission.food_recipes:
            actions.append(f"c_{k}")
        # (3) Every "cook" action "cook_<food_name>":
        for k in submission.cooking_recipes:
            actions.append(f"cook_{k}")
        # (4) Terminal "present_gift"
        actions.append("present_gift")
        return actions

    def state_to_tensor(self, state_tuple):
        """
        Convert a state_tuple like (('beef',1), ('cooked_porkchop',2), …)
        into a 1×num_item_types tensor of normalized counts.
        """
        vec = torch.zeros(self.num_item_types, dtype=torch.float, device=self.device)
        for (item_name, count) in state_tuple:
            idx = self.item_list.index(item_name)  # ←==== UPDATED
            vec[idx] = count / float(self.inventory_limit)
        return vec

    def clear_inventory(self):
        """Resets the inventory in case of a new attempt to fetch. """
        self.inventory = defaultdict(lambda: 0, {})
        self.num_items_in_inv = 0

    def get_crafting_options(self):
        import copy
        craft_opt = []
        inventory_items = []
        for item, count in self.inventory.items():
            for j in range(count):
                inventory_items.append(item)

        for item, recipe in food_recipes.items():
            t_inventory_items = copy.deepcopy(inventory_items)
            inter = []
            for i in recipe:
                if i in t_inventory_items:
                    inter.append(i)
                    t_inventory_items.remove(i)
            if len(inter) == len(recipe):
                craft_opt.append(item)

        # print(f"craft_opt: {craft_opt}")
        return craft_opt
    
    @staticmethod
    def get_obj_locations(agent_host):
        """Queries for the object's location in the world.

        As a side effect it also returns morgan's location.
        """
        nearyby_obs = {}
        while True:
            world_state = agent_host.getWorldState()
            # print(f"world_state: {world_state}")
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                # print(f"ob: {ob}")
                for ent in  ob['entities']:
                    name = ent['name']
                    # if name != 'morgan':
                    nearyby_obs[name] = (ent['yaw'], ent['x'], ent['z'])

                return nearyby_obs

    def was_item_picked(self, agent_host, item):
        """Goes over the inventory observation and check if the item was picked. """
        prev_item_count = self.inventory[item]
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)

                for i in range(9):
                    key = 'InventorySlot_%d_item' % i
                    if key in ob:
                        inv_item = ob[key]
                        inv_counts = ob['InventorySlot_%d_size' % i]

                        if inv_item == item and inv_counts > prev_item_count:
                            return True
                    else:
                        break

            return False

    def teleport(self, agent_host, teleport_x, teleport_z):
        """Directly teleport to a specific position."""
        tp_command = "tp " + str(teleport_x)+ " 227 " + str(teleport_z)
        agent_host.sendCommand(tp_command)
        good_frame = False
        start = timer()
        while not good_frame:
            world_state = agent_host.getWorldState()
            if not world_state.is_mission_running:
                print("Mission ended prematurely - error.")
                exit(1)
            if not good_frame and world_state.number_of_video_frames_since_last_state > 0:
                frame_x = world_state.video_frames[-1].xPos
                frame_z = world_state.video_frames[-1].zPos
                if math.fabs(frame_x - teleport_x) < 0.001 and math.fabs(frame_z - teleport_z) < 0.001:
                    good_frame = True
                    end_frame = timer()

    def move_to(self, agent_host, target_x, target_z, timeout=30):
        """Move to absolute target coordinates"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get current position
            obj_locs = self.get_obj_locations(agent_host)
            if 'morgan' not in obj_locs:
                return False
            
            _, curr_x, curr_z = obj_locs['morgan']
            
            # Calculate distance to target
            distance = math.sqrt((target_x - curr_x)**2 + (target_z - curr_z)**2)
            
            # If we're close enough, stop
            if distance < 0.5:
                agent_host.sendCommand("move 0")
                agent_host.sendCommand("sprint 0")
                print(f"[MOVE] Reached target. Final distance: {distance:.2f}")
                return True
            
            # Calculate direction to target
            dx = target_x - curr_x
            dz = target_z - curr_z
            angle_to_target = math.degrees(math.atan2(-dx, dz)) % 360
            
            # Face the target direction
            agent_host.sendCommand(f"setYaw {angle_to_target}")
            
            # Move forward
            agent_host.sendCommand("move 1")
            agent_host.sendCommand("sprint 1")
            
            # Small delay to allow movement
            time.sleep(0.1)
        
        # If we get here, we timed out
        agent_host.sendCommand("move 0")
        agent_host.sendCommand("sprint 0")
        print("[MOVE] Timed out before reaching target")
        return False

    def fetch_item(self, agent_host, item_to_pick):  
        if self.num_items_in_inv > inventory_limit:
            print(f"[ACTION] Cannot fetch {item_to_pick}: Inventory full ({self.num_items_in_inv}/{inventory_limit})")
            return
            
        print(f"[ACTION] Attempting to fetch: {item_to_pick}")
        
        # Get current and target positions
        obj_locs = self.get_obj_locations(agent_host)
        if 'morgan' not in obj_locs or item_to_pick not in obj_locs:
            print(f"[ERROR] Cannot locate Morgan or {item_to_pick}")
            return
            
        my_yaw, my_x, my_z = obj_locs['morgan']
        obj_yaw, obj_x, obj_z = obj_locs[item_to_pick]
        
        # Use A* pathfinding to move to the item
        if hasattr(self, 'pathfinder'):
            print(f"[PATH] Moving to {item_to_pick} using pathfinding")
            # Convert positions to grid coordinates
            start_pos = (int(round(my_x)), int(round(my_z)))
            goal_pos = (int(round(obj_x)), int(round(obj_z)))
            
            # Update pathfinder's world state
            self.pathfinder.update_world_state(agent_host)
            
            # Find and execute path
            path = self.pathfinder.a_star(start_pos, goal_pos)
            if path:
                self.execute_path(agent_host, path)
            else:
                print(f"[PATH] No path found to {item_to_pick}, using direct movement")
                self.move_to(agent_host, obj_x, obj_z)
        else:
            print(f"[PATH] Moving to {item_to_pick} using direct movement")
            self.move_to(agent_host, obj_x, obj_z)
            
        # Wait for item pickup
        while True:
            if self.was_item_picked(agent_host, item_to_pick) or item_to_pick not in obj_locs:
                break
                
        # Return to starting position using A*
        if hasattr(self, 'pathfinder'):
            print("[PATH] Returning to center")
            start_pos = (int(round(obj_x)), int(round(obj_z)))
            goal_pos = (0, 0)  # Return to center
            path = self.pathfinder.a_star(start_pos, goal_pos)
            if path:
                self.execute_path(agent_host, path)
            else:
                print("[PATH] No path to center, using direct movement")
                self.move_to(agent_host, 0.5, 0.5)
        else:
            self.move_to(agent_host, 0.5, 0.5)

        self.inventory[item_to_pick] += 1
        self.num_items_in_inv += 1
        print(f"[SUCCESS] Added {item_to_pick} to inventory. Current items: {dict(self.inventory)}")
        time.sleep(0.1)

    def get_block_position(self, agent_host, block_type, grid_name="floor_all", x_range=21, y_range=1, z_range=21, grid_min_x=-10, grid_min_z=-10, max_wait_seconds=5):
        """Get absolute position of a specific block type in the grid"""
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                try:
                    obs = json.loads(world_state.observations[-1].text)
                    if grid_name in obs:
                        grid = obs[grid_name]
                        
                        # Search for matching blocks in the grid
                        for idx, block in enumerate(grid):
                            if block == block_type or block == f"lit_{block_type}":
                                # Convert grid index to absolute world coordinates
                                # Grid is ordered by x first, then z
                                x_idx = idx % x_range
                                z_idx = idx // x_range
                                
                                # Convert to absolute world coordinates
                                world_x = grid_min_x + x_idx
                                world_z = grid_min_z + z_idx
                                
                                print(f"[DEBUG] Found {block_type} at absolute coordinates ({world_x}, {world_z})")
                                return world_x, world_z
                                
                except Exception as e:
                    print(f"[ERROR] Error finding {block_type}:", e)
            time.sleep(0.1)

        print(f"[WARN] Could not find block '{block_type}' in grid '{grid_name}' within {max_wait_seconds} seconds")
        return None, None

    def can_cook(self, agent_host, threshold=2):
        furnace_x, furnace_z = self.get_block_position(agent_host, "furnace")
        if furnace_x is None:
            return False

        obj_locs = self.get_obj_locations(agent_host)
        if 'morgan' not in obj_locs:
            return False

        _, agent_x, agent_z = obj_locs['morgan']
        distance = math.sqrt((furnace_x - agent_x)**2 + (furnace_z - agent_z)**2)
        print(f"[DEBUG] Agent at absolute ({agent_x:.1f}, {agent_z:.1f}), furnace at absolute ({furnace_x}, {furnace_z}), distance: {distance:.1f}")
        
        # If we're not close enough, try to approach
        if distance > threshold:
            print("[DEBUG] Not close enough to furnace, attempting approach")
            if self.approach_furnace(agent_host):
                # Recheck distance after approach
                obj_locs = self.get_obj_locations(agent_host)
                if 'morgan' in obj_locs:
                    _, agent_x, agent_z = obj_locs['morgan']
                    distance = math.sqrt((furnace_x - agent_x)**2 + (furnace_z - agent_z)**2)
                    print(f"[DEBUG] After approach: distance = {distance:.1f}")
                    return distance <= threshold
            return False
            
        return True

    def cook_item(self, agent_host, cooked_item):
        print(f"\n[ACTION] Attempting to cook: {cooked_item}")
        
        # Check ingredients first
        if cooked_item not in submission.cooking_recipes:
            print(f"[ERROR] No recipe found for {cooked_item}")
            return False

        ingredients = submission.cooking_recipes[cooked_item]
        print(f"[INFO] Recipe requires: {ingredients}")
        
        # Verify ingredients
        for item in ingredients:
            if self.inventory[item] < ingredients.count(item):
                print(f"[ERROR] Not enough {item} (have {self.inventory[item]}, need {ingredients.count(item)})")
                return False
        print("[INFO] All ingredients available")

        # Move to furnace if needed
        if not self.can_cook(agent_host):
            print("[PATH] Moving to furnace")
            furnace_x, furnace_z = self.get_block_position(agent_host, "furnace")
            if furnace_x is not None:
                self.move_to(agent_host, furnace_x, furnace_z)
                time.sleep(0.5)  # Wait for movement to complete
                
                # Check again if we're close enough
                if not self.can_cook(agent_host):
                    print("[ERROR] Failed to reach furnace")
                    return False
            else:
                print("[ERROR] Could not find furnace")
                return False
        
        print("[ACTION] Exchanging items")
        print(f"[INFO] Item: {cooked_item}")
        
        # Perform the cooking operation
        try:
            # Combine the ingredients to cook
            agent_host.sendCommand(f"craft {cooked_item}")
            time.sleep(0.1)  # Small delay to allow crafting
            
            # Update our inventory tracking
            for item in ingredients:
                self.inventory[item] -= 1
                self.num_items_in_inv -= 1
                print(f"[INFO] Used 1x {item}")

            self.inventory[cooked_item] += 1
            self.num_items_in_inv += 1
            print(f"[SUCCESS] Cooked {cooked_item}. Current items: {dict(self.inventory)}")
            return True
            
        except RuntimeError as e:
            print(f"[ERROR] Failed to cook {cooked_item}: {e}")
            return False

    def can_craft(self, agent_host, threshold=1.5):
        table_x, table_z = self.get_block_position(agent_host, "crafting_table")
        if table_x is None:
            return False
        obj_locs = self.get_obj_locations(agent_host)
        if 'morgan' not in obj_locs:
            return False
        _, agent_x, agent_z = obj_locs['morgan']
        # Calculate distance from agent to crafting table center
        distance = math.sqrt((table_x - agent_x)**2 + (table_z - agent_z)**2)
        print(f"[DEBUG] Agent at absolute ({agent_x:.1f}, {agent_z:.1f}), crafting table at absolute ({table_x}, {table_z}), distance: {distance:.1f}")
        
        # If we're not close enough, try to approach
        if distance > threshold:
            print("[DEBUG] Not close enough to crafting table, attempting approach")
            if self.approach_crafting_table(agent_host):
                # Recheck distance after approach
                obj_locs = self.get_obj_locations(agent_host)
                if 'morgan' in obj_locs:
                    _, agent_x, agent_z = obj_locs['morgan']
                    distance = math.sqrt((table_x - agent_x)**2 + (table_z - agent_z)**2)
                    print(f"[DEBUG] After approach: distance = {distance:.1f}")
                    return distance <= threshold
            return False
            
        return True

    def craft_item(self, agent_host, item):
        print(f"\n[ACTION] Attempting to craft: {item}")
        
        # Check recipe first
        if item not in submission.food_recipes:
            print(f"[ERROR] No recipe found for {item}")
            return

        ingredients = submission.food_recipes[item]
        print(f"[INFO] Recipe requires: {ingredients}")
        
        # Verify ingredients
        for item_needed in ingredients:
            if self.inventory[item_needed] < ingredients.count(item_needed):
                print(f"[ERROR] Not enough {item_needed} (have {self.inventory[item_needed]}, need {ingredients.count(item_needed)})")
                return
        print("[INFO] All ingredients available")

        # Move to crafting table if needed
        if not self.can_craft(agent_host):
            print("[PATH] Moving to crafting table")
            table_x, table_z = self.get_block_position(agent_host, "crafting_table")
            if table_x is not None:
                self.move_to(agent_host, table_x, table_z)
                time.sleep(0.5)  # Wait for movement to complete
                
                # Check again if we're close enough
                if not self.can_craft(agent_host):
                    print("[ERROR] Failed to reach crafting table")
                    return
            else:
                print("[ERROR] Could not find crafting table")
                return

        print("[ACTION] Starting crafting process")
        # Remove ingredients
        for item_needed in ingredients:
            self.inventory[item_needed] -= 1
            self.num_items_in_inv -= 1
            print(f"[INFO] Used 1x {item_needed}")

        # Craft the item
        agent_host.sendCommand(f'craft {item}')
        self.inventory[item] += 1
        self.num_items_in_inv += 1
        time.sleep(0.25)
        print(f"[SUCCESS] Crafted {item}. Current items: {dict(self.inventory)}")

    def execute_path(self, agent_host, path):
        """Execute a path by moving through each waypoint"""
        if not path or len(path) < 2:
            print("[PATH] Path too short to execute")
            return False
            
        # print(f"[PATH] Executing path with {len(path)} waypoints")
        
        # Move through each waypoint in the path
        for i, (target_x, target_z) in enumerate(path[1:], 1):
            # print(f"[PATH] Moving to waypoint {i}/{len(path)-1}: ({target_x}, {target_z})")
            
            # Add a small offset to help the agent reach the center of cells
            adjusted_x = target_x + 0.5
            adjusted_z = target_z + 0.5
            
            self.move_to(agent_host, adjusted_x, adjusted_z)
            time.sleep(0.1)  # Small delay between moves
        
        print("[PATH] Path execution completed")
        return True

    def present_gift(self, agent_host):
        """Calculates the reward points for the current inventory.

        Args
            agent_host: the host object

        Returns
            reward:     <float> current reward from world state
        """
        current_r = 0
        
        print("\nPresenting items:")
        for item, counts in self.inventory.items():
            if counts > 0:  # Only show items that are actually in inventory
                item_reward = rewards_map[item] * counts
                current_r += item_reward
                print(f"  {counts}x {item:<15} -> {item_reward:>4} points")
        print(f"Total reward: {current_r}")

        agent_host.sendCommand('quit')
        return current_r

    @staticmethod
    def is_solution(reward):
        """If the reward equals to the maximum reward possible returns True, False otherwise. """
        return submission.is_solution(reward)

    def get_possible_actions(self, agent_host, is_first_action=False):
        action_list = []
        if not is_first_action:
            action_list = ['present_gift']

        craft_opt = self.get_crafting_options()
        if len(craft_opt) > 0:
            action_list.extend(['c_%s' % craft_item for craft_item in craft_opt])

        if self.num_items_in_inv < inventory_limit:
            nearby_obj = self.get_obj_locations(agent_host)
            if len(nearby_obj) > 1:
                action_list.extend([item for item in nearby_obj.keys() if item != 'morgan'])

        for cooked_item, ingredients in cooking_recipe.items():
            if all(self.inventory[i] >= ingredients.count(i) for i in ingredients):
                action_list.append(f"cook_{cooked_item}")

        return action_list

    def get_curr_state(self):
        """Creates a unique identifier for a state.

        The state is defined as the items in the agent inventory. Notice that the state has to be sorted -- otherwise
        differnt order in the inventory will be different states.
        """
        return submission.get_curr_state(self.inventory.items())

    def choose_action(self, curr_state, possible_actions, is_first_action=False):
        if not self.use_dqn:
            # Fallback to existing tabular logic:
            if curr_state not in self.q_table:
                self.q_table[curr_state] = {}
            for action in possible_actions:
                if action not in self.q_table[curr_state]:
                    self.q_table[curr_state][action] = 0
            return submission.choose_action(curr_state, possible_actions, self.epsilon, self.q_table)

        # ----- DQN ε-greedy policy -----
        state_tensor = self.state_to_tensor(curr_state).unsqueeze(0)   # shape: (1, num_item_types)
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # Get all Q-values from policy_net
        with torch.no_grad():
            q_values_all = self.policy_net(state_tensor).squeeze(0)     # shape: (num_actions,)
        # Build legal‐action mask
        legal_indices = [self.all_actions.index(a) for a in possible_actions]
        illegal_indices = list(set(range(self.num_actions)) - set(legal_indices))

        if sample < eps_threshold:
            # random legal action
            action_idx = random.choice(legal_indices)
        else:
            # mask illegal actions to −∞ so they're never chosen
            for idx in illegal_indices:
                q_values_all[idx] = float("-inf")
            action_idx = torch.argmax(q_values_all).item()

        action_str = self.all_actions[action_idx]
        return action_idx, action_str

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        state_batch      = torch.stack([self.state_to_tensor(s)  for s  in batch[0]]).to(self.device)
        action_batch     = torch.tensor(batch[1], dtype=torch.long, device=self.device)
        reward_batch     = torch.tensor(batch[2], dtype=torch.float, device=self.device)
        next_state_batch = torch.stack([self.state_to_tensor(s2) for s2 in batch[3]]).to(self.device)
        done_batch       = torch.tensor(batch[4], dtype=torch.float, device=self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(dim=1)[0]
            target_q      = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def find_block_position(self, agent_host, block_type, max_wait_seconds=5):
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                try:
                    obs = json.loads(world_state.observations[-1].text)
                    if 'floor_all' in obs:
                        grid = obs['floor_all']
                        
                        # Search for matching blocks in the grid
                        for idx, block in enumerate(grid):
                            if block == block_type or block == f"lit_{block_type}":
                                # Convert grid index to absolute world coordinates
                                # Grid is ordered by x first, then z
                                x_idx = idx % 21  # Grid size is 21x21
                                z_idx = idx // 21
                                
                                # Convert to absolute world coordinates
                                world_x = -10 + x_idx  # Grid starts at -10
                                world_z = -10 + z_idx
                                
                                print(f"[DEBUG] Found {block_type} at absolute coordinates ({world_x}, {world_z})")
                                return world_x, world_z
                                
                except Exception as e:
                    print(f"[ERROR] Error finding {block_type}:", e)
            time.sleep(0.1)

        print(f"[WARN] Could not find block '{block_type}' in grid")
        return None, None

    def act(self, agent_host, action):
        print(f"\n[STEP] Executing action: {action}")
        if action == 'present_gift':
            return self.present_gift(agent_host)
        elif action.startswith('c_'):
            self.craft_item(agent_host, action[2:])
        elif action.startswith('cook_'):
            self.cook_item(agent_host, action[len('cook_'):])
        else:
            self.fetch_item(agent_host, action)
        return 0

    def update_q_table(self, tau, S, A, R, T):
        """Performs relevant updates for state tau.

        Args
            tau: <int>  state index to update
            S:   <dequqe>   states queue
            A:   <dequqe>   actions queue
            R:   <dequqe>   rewards queue
            T:   <int>      terminating state index
        """
        curr_s, curr_a, curr_r = S.popleft(), A.popleft(), R.popleft()
        G = sum([self.gamma ** i * R[i] for i in range(len(S))])
        if tau + self.n < T:
            G += self.gamma ** self.n * self.q_table[S[-1]][A[-1]]

        old_q = self.q_table[curr_s][curr_a]
        self.q_table[curr_s][curr_a] = old_q + self.alpha * (G - old_q)

    def best_policy(self, agent_host):
        """Reconstructs the best action list according to the greedy policy. """
        if not self.use_dqn:
            # Fallback to original tabular version
            ...
            return

        # --- DQN greedy rollout (ε=0) ---
        self.clear_inventory()
        state = self.get_curr_state()
        policy = []
        while True:
            possible_actions = self.get_possible_actions(agent_host, is_first_action=(len(policy) == 0))
            # Get Q-values
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                q_all = self.policy_net(state_tensor).squeeze(0)
            # Mask illegal actions
            legal_indices = [self.all_actions.index(a) for a in possible_actions]
            illegal_indices = list(set(range(self.num_actions)) - set(legal_indices))
            for idx in illegal_indices:
                q_all[idx] = float("-inf")
            action_idx = torch.argmax(q_all).item()
            action_str = self.all_actions[action_idx]
            policy.append(action_str)
            reward = self.act(agent_host, action_str)
            if action_str == "present_gift":
                break
            state = self.get_curr_state()

        print("Best DQN policy:", policy, "Reward:", reward)
        return submission.is_solution(reward)

    def run(self, agent_host):
        if not self.use_dqn:
            # Fallback to original tabular run
            S, A, R = deque(), deque(), deque()
            present_reward = 0
            done_update = False
            total_reward = 0.0  # Track total reward for the episode
            
            while not done_update:
                s0 = self.get_curr_state()
                possible_actions = self.get_possible_actions(agent_host, True)
                a0 = self.choose_action(s0, possible_actions, self.epsilon)
                S.append(s0)
                A.append(a0)
                R.append(0)

                T = sys.maxsize
                for t in range(sys.maxsize):
                    time.sleep(0.1)
                    if t < T:
                        current_r = self.act(agent_host, A[-1])
                        R.append(current_r)
                        total_reward += current_r  # Accumulate reward

                        if A[-1] == "present_gift":
                            # Terminating state
                            T = t + 1
                            S.append('Term State')
                            present_reward = current_r
                        else:
                            s = self.get_curr_state()
                            S.append(s)
                            possible_actions = self.get_possible_actions(agent_host)
                            next_a = self.choose_action(s, possible_actions, self.epsilon)
                            A.append(next_a)

                    tau = t - self.n + 1
                    if tau >= 0:
                        self.update_q_table(tau, S, A, R, T)

                    if tau == T - 1:
                        while len(S) > 1:
                            tau = tau + 1
                            self.update_q_table(tau, S, A, R, T)
                        done_update = True
                        break
            return

        # --- New DQN training loop ---
        state = self.get_curr_state()
        total_reward = 0.0

        for t in range(MAX_STEPS_PER_EPISODE):
            possible_actions = self.get_possible_actions(agent_host, is_first_action=(t == 0))
            action_idx, action_str = self.choose_action(state, possible_actions, is_first_action=(t == 0))
            reward = self.act(agent_host, action_str)
            total_reward += reward

            next_state = self.get_curr_state()
            done = (action_str == "present_gift")

            self.memory.push(state, action_idx, reward, next_state, done)
            state = next_state

            loss = self.optimize_model()
            if loss is not None:
                self.loss_history.append(loss.item())

            if done:
                break

        # ←–– Episode has ended here ––→

        # 1) Record total_reward
        self.episode_rewards.append(total_reward)

        # 2) Print a per‐episode summary:
        print(
            f"Episode {self.episode_counter+1:4d} | "
            f"TotalReward: {total_reward:.1f} | "
            f"Epsilon: {self.epsilon:.3f} | "
            f"RecentAvgLoss: {np.mean(self.loss_history[-10:]):.4f}"
        )

        # 3) Update target network every TARGET_UPDATE episodes:
        prev_count = getattr(self, "episode_counter", 0)
        self.episode_counter = prev_count + 1
        if self.episode_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def approach_furnace(self, agent_host):
        """Make a precise approach to the furnace using absolute coordinates"""
        furnace_x, furnace_z = self.get_block_position(agent_host, "furnace")
        if furnace_x is None:
            print("[ERROR] Could not find furnace")
            return False
        
        # Get current position
        obj_locs = self.get_obj_locations(agent_host)
        if 'morgan' not in obj_locs:
            print("[ERROR] Cannot locate agent position")
            return False
        
        _, curr_x, curr_z = obj_locs['morgan']
        
        # Calculate vector to furnace
        dx = furnace_x - curr_x
        dz = furnace_z - curr_z
        distance = math.sqrt(dx*dx + dz*dz)
        
        # If we're already close enough, just face the furnace
        if distance <= 2:
            angle_to_furnace = math.degrees(math.atan2(-dx, dz)) % 360
            agent_host.sendCommand(f"setYaw {angle_to_furnace}")
            time.sleep(0.2)
            return True
        
        # Calculate target position 1.5 blocks away from furnace
        target_x = furnace_x - (dx * 1.5/distance)
        target_z = furnace_z - (dz * 1.5/distance)
        
        # Move to target position
        success = self.move_to(agent_host, target_x, target_z, timeout=5)
        if not success:
            print("[WARN] Failed to reach ideal furnace position")
            return False
        
        # Face the furnace
        angle_to_furnace = math.degrees(math.atan2(-(furnace_x - target_x), furnace_z - target_z)) % 360
        agent_host.sendCommand(f"setYaw {angle_to_furnace}")
        time.sleep(0.2)
        
        return True
        
    def approach_crafting_table(self, agent_host):
        """Make a precise approach to the crafting table using absolute coordinates"""
        table_x, table_z = self.get_block_position(agent_host, "crafting_table")
        if table_x is None:
            print("[ERROR] Could not find crafting table")
            return False
        
        # Get current position
        obj_locs = self.get_obj_locations(agent_host)
        if 'morgan' not in obj_locs:
            print("[ERROR] Cannot locate agent position")
            return False
        
        _, curr_x, curr_z = obj_locs['morgan']
        
        # Calculate vector to crafting table
        dx = table_x - curr_x
        dz = table_z - curr_z
        distance = math.sqrt(dx*dx + dz*dz)
        
        # If we're already close enough, just face the table
        if distance <= 1.5:
            angle_to_table = math.degrees(math.atan2(-dx, dz)) % 360
            agent_host.sendCommand(f"setYaw {angle_to_table}")
            time.sleep(0.2)
            return True
        
        # Calculate target position 1 block away from table
        target_x = table_x - (dx * 1.0/distance)
        target_z = table_z - (dz * 1.0/distance)
        
        # Move to target position
        success = self.move_to(agent_host, target_x, target_z, timeout=5)
        if not success:
            print("[WARN] Failed to reach ideal crafting table position")
            return False
        
        # Face the table
        angle_to_table = math.degrees(math.atan2(-(table_x - target_x), table_z - target_z)) % 360
        agent_host.sendCommand(f"setYaw {angle_to_table}")
        time.sleep(0.2)
        
        return True
