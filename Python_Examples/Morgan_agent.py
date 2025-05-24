from __future__ import division
import numpy as np

import MalmoPython
import os
import random
import sys
import time
import json
import random
import math
import errno
import hunger_learner_helper as submission
from collections import defaultdict, deque
from timeit import default_timer as timer

items=submission.items
inventory_limit = 3
food_recipes = submission.food_recipes
rewards_map = submission.rewards_map
cooking_recipe = submission.cooking_recipes

class Morgan(object):
    def __init__(self, alpha=0.3, gamma=1, n=1):
        """Constructing an RL agent.

        Args
            alpha:  <float>  learning rate      (default = 0.3)
            gamma:  <float>  value decay rate   (default = 1)
            n:      <int>    number of back steps to update (default = 1)
        """
        self.epsilon = 0.2  # chance of taking a random action instead of the best
        self.q_table = {}
        self.n, self.alpha, self.gamma = n, alpha, gamma
        self.inventory = defaultdict(lambda: 0, {})
        self.num_items_in_inv = 0

    def clear_inventory(self):
        """Resets the inventory in case of a new attempt to fetch. """
        self.inventory = defaultdict(lambda: 0, {})
        self.num_items_in_inv = 0

    def get_crafting_options(self):
        """Returns the objects that can be crafted from the inventory. """
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

        return craft_opt
    

    @staticmethod
    def get_obj_locations(agent_host):
        """Queries for the object's location in the world.

        As a side effect it also returns morgan's location.
        """
        nearyby_obs = {}
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
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

    def move_to(self, agent_host, target_x, target_z):
        obj_locs = self.get_obj_locations(agent_host)
        _, curr_x, curr_z = obj_locs['morgan']

        dx = target_x - curr_x
        dz = target_z - curr_z
        angle_to_target = math.degrees(math.atan2(-dx, dz)) % 360

        # Face the target
        agent_host.sendCommand(f"setYaw {angle_to_target}")
        time.sleep(0.2)

        # Start sprinting and moving
        agent_host.sendCommand("sprint 1")
        agent_host.sendCommand("move 1")

        while True:
            # print("test")
            time.sleep(0.1)
            obj_locs = self.get_obj_locations(agent_host)
            _, curr_x, curr_z = obj_locs['morgan']
            dx = target_x - curr_x
            dz = target_z - curr_z
            distance = math.sqrt(dx**2 + dz**2)

            if distance <= 0.5:
                print(f"Hunger: {self.get_hunger_level(agent_host)}")
                break


        agent_host.sendCommand("move 0")
        agent_host.sendCommand("sprint 0")

    def get_hunger_level(self, agent_host):
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            obs = json.loads(world_state.observations[-1].text)
            return obs.get("Food", 20)

    def is_near_block(self, agent_host, block_type, threshold=1.5):
        obj_locs = self.get_obj_locations(agent_host)
        _, agent_x, agent_z = obj_locs['morgan']
        for name, (yaw, x, z) in obj_locs.items():
            if name == block_type:
                dist = math.sqrt((x - agent_x)**2 + (z - agent_z)**2)
                return dist <= threshold
        return False


    def fetch_item(self, agent_host, item_to_pick):
        """Finds the object in the world and picks it up (by teleporting to it).

        Will not pick up the item if morgan has more than 3 items in his mouth :)
        """
        if self.num_items_in_inv > inventory_limit:
            return
        # teleport
        obj_locs = self.get_obj_locations(agent_host)
        my_yaw, my_x, my_z = obj_locs['morgan']
        obj_yaw, obj_x, obj_z = obj_locs[item_to_pick]
        self.teleport(agent_host, obj_x, obj_z)
        # self.move_to(agent_host, obj_x, obj_z)
        time.sleep(0.1)  # Letting the host pick up on the things that were picked up
        while True:
            if self.was_item_picked(agent_host, item_to_pick) or item_to_pick not in obj_locs:
                break
        self.teleport(agent_host, 0.5, 0.5)
        time.sleep(0.1)  # Letting the host pick up on the things that were picked up

        self.inventory[item_to_pick] += 1
        self.num_items_in_inv += 1

    def craft_item(self, agent_host, item):
        """Creates item from the current inventory.

        Raised assertion error if any item is missing and will stop the whole process.
        (so don't call it unless you're sure you have all the items, that's why the craft_option
        method is for :) )

        It replaces the item in the inventory dictionary.
        """

        if not self.is_near_block(agent_host, "crafting_table"):
            print("Not near crafting table!")
            return
        
        items_needed = food_recipes[item]
        for item_needed in items_needed:
            self.inventory[item_needed] -= 1
            self.num_items_in_inv -= 1
            if self.inventory[item_needed] < 0:
                raise AssertionError('Missing items for crafting: %s in %s' % (item_needed, str(self.inventory_items)))

        agent_host.sendCommand('craft %s' % item)
        self.inventory[item] += 1
        self.num_items_in_inv += 1
        time.sleep(0.25)

    def cook_item(self, agent_host, cooked_item):
        if not self.is_near_block(agent_host, "furnace"):
            print("Not near furnace!")
            return
        
        ingredients = cooking_recipe[cooked_item]
        for item in ingredients:
            self.inventory[item] -= 1
            self.num_items_in_inv -= 1

        self.inventory[cooked_item] += 1
        self.num_items_in_inv += 1
        time.sleep(0.5)  # Optional: simulate cooking time


    def present_gift(self, agent_host):
        """Calculates the reward points for the current inventory.

        Args
            agent_host: the host object

        Returns
            reward:     <float> current reward from world state
        """
        current_r = 0
        #time.sleep(0.1)

        for item, counts in self.inventory.items():
            current_r += rewards_map[item] * counts

        agent_host.sendCommand('quit')
        #time.sleep(0.25)
        return current_r

    @staticmethod
    def is_solution(reward):
        """If the reward equals to the maximum reward possible returns True, False otherwise. """
        return submission.is_solution(reward)

    def get_possible_actions(self, agent_host, is_first_action=False):
        """Returns all possible actions that can be done at the current state. """
        action_list = []
        if not is_first_action:
            # Not allowing morgan to come back empty.
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

    def choose_action(self, curr_state, possible_actions, eps):
        """Chooses an action according to eps-greedy policy. """
        if curr_state not in self.q_table:
            self.q_table[curr_state] = {}
        for action in possible_actions:
            if action not in self.q_table[curr_state]:
                self.q_table[curr_state][action] = 0

        return submission.choose_action(curr_state, possible_actions, eps, self.q_table)

    @staticmethod
    def get_obj_locations(agent_host):
        nearyby_obs = {}
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                for ent in ob['entities']:
                    name = ent['name']
                    nearyby_obs[name] = (ent['yaw'], ent['x'], ent['z'])
                return nearyby_obs

    def find_block_position(self, agent_host, block_type, max_wait_seconds=5):
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                try:
                    print(world_state.observations[-1])
                    obs = json.loads(world_state.observations[-1].text)
                    if 'floor_all' in obs:
                        grid = obs['floor_all']
                        
                        obj_locs = self.get_obj_locations(agent_host)
                        _, x_pos, z_pos = obj_locs['morgan']

                        x_len = 81  # -40 to 40
                        y_len = 2   # 227 to 228
                        z_len = 81

                        for idx, block in enumerate(grid):
                            if block in [block_type, f"lit_{block_type}"]:
                                x_idx = idx % x_len
                                y_idx = (idx // x_len) % y_len
                                z_idx = idx // (x_len * y_len)

                                dx = x_idx - 40
                                dz = z_idx - 40

                                return x_pos + dx, z_pos + dz
                except Exception as e:
                    print("Error parsing observation:", e)
            time.sleep(0.1)

        print(f"[WARN] Could not find block type '{block_type}' within {max_wait_seconds}s")
        return None, None


    def act(self, agent_host, action):
        print(action + ",", end=" ")
        if action == 'present_gift':
            return self.present_gift(agent_host)
        elif action.startswith('c_'):
            # x, z = self.find_block_position(agent_host, "crafting_table")
            # if x is not None:
            #     self.move_to(agent_host, x, z)
            self.craft_item(agent_host, action[2:])
        elif action.startswith('cook_'):
            # x, z = self.find_block_position(agent_host, "furnace")
            # if x is not None:
            #     self.move_to(agent_host, x, z)
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
        self.clear_inventory()
        policy = []
        current_r = 0
        is_first_action = True
        next_a = ""
        while next_a != "present_gift":
            curr_state = self.get_curr_state()
            possible_actions = self.get_possible_actions(agent_host, is_first_action)
            next_a = self.choose_action(curr_state, possible_actions, 0)
            policy.append(next_a)
            is_first_action = False
            current_r = self.act(agent_host, next_a)
        print(' with reward %.1f' % (current_r))
        return self.is_solution(current_r)
        #print 'Best policy so far is %s with reward %.1f' % (policy, current_r)

    def run(self, agent_host):
        """Learns the process to compile the best gift for dad. """
        S, A, R = deque(), deque(), deque()
        present_reward = 0
        done_update = False
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

                    if A[-1] == "present_gift":
                        # Terminating state
                        T = t + 1
                        S.append('Term State')
                        present_reward = current_r
                        print("Reward:", present_reward)
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