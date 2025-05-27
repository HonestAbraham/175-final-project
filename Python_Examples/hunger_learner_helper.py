import random
import heapq

items = [
    # 'beef',
    # 'porkchop',
    # # 'fish',
    # 'rabbit',
    # 'coal',

    'pumpkin',
    'planks',
    'planks'
]


food_recipes = {
#         'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
        'pumpkin_seeds': ['pumpkin'],
        'bowl': ['planks', 'planks'],
#         'mushroom_stew': ['bowl', 'red_mushroom']
    }



cooking_recipes = {
    'cooked_beef': ['coal', 'beef'],
    'cooked_porkchop': ['coal', 'porkchop'],
    'cooked_rabbit': ['coal', 'rabbit']
}


rewards_map = {
    'beef': 3,
    'porkchop': 3,
    'rabbit': 3,
    'coal': 0,
    'cooked_beef': 8,
    'cooked_porkchop': 8,
    'cooked_rabbit': 5,
    
    'pumpkin_seeds': 1,
    'pumpkin': 0,
    'bowl': 1,
    'planks': 0
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


def save_q_table_to_file(q_table, filename="q_table_output.txt"):
    with open(filename, 'w') as f:
        for state, actions in q_table.items():
            f.write(f"State: {state}\n")
            for action, value in actions.items():
                f.write(f"  Action: {action.ljust(20)} Q-value: {value:.2f}\n")
            f.write("\n")

def is_solution(reward):
    return reward == 11


def is_walkable(x, y, z, grid, grid_min_x, grid_min_y, grid_min_z, x_len, y_len, z_len):
    def get_block(x, y, z):
        gx = x - grid_min_x
        gy = y - grid_min_y
        gz = z - grid_min_z

        if 0 <= gx < x_len and 0 <= gy < y_len and 0 <= gz < z_len:
            index = gx + x_len * (gy + y_len * gz)
            return grid[index]
        return 'air'  # default to walkable

    block_below = get_block(x, y - 1, z)
    block_feet = get_block(x, y, z)
    block_head = get_block(x, y + 1, z)

    print(f"block below: {block_below}")
    print(f"block feet: {block_feet}")
    print(f"block head: {block_head}")

    # solid_blocks = {'stone', 'dirt', 'grass', 'wood', 'crafting_table', 'furnace'}
    air_blocks = {'air'}

    return (block_below not in air_blocks and
            block_feet in air_blocks and
            block_head in air_blocks)


def pathfindingSearch2D(start_x, start_y, start_z, dest_x, dest_y, dest_z, grid, grid_min_x, grid_min_y, grid_min_z, x_len, y_len, z_len):
    """
    A* 2D pathfinding (flat y-level).
    Returns: a list of (x, z) positions from start to goal (inclusive)
    """
    start = (start_x, start_z)
    goal = (dest_x, dest_z)

    print(f"Start: {start}")
    print(f"goal: {goal}")

    def calculate_heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(pos):
        x, z = pos
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dz in directions:
            nx, nz = x + dx, z + dz
            if is_walkable(nx, start_y, nz, grid, grid_min_x, grid_min_y, grid_min_z, x_len, y_len, z_len):
                yield (nx, nz)

    open_set = []
    heapq.heappush(open_set, (calculate_heuristic(start, goal), 0, start, [start]))
    visited = set()
    

    while open_set:
        f, cost, current, path = heapq.heappop(open_set)
        
        print(f"{f}, {cost}, {current}, {path}")

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        print(f"current: {current}")
        print(f"neighbors: {get_neighbors(current)}")

        for neighbor in get_neighbors(current):
            print(f"here though?")
            if neighbor in visited:
                continue
            heapq.heappush(open_set, (
                cost + 1 + calculate_heuristic(neighbor, goal),
                cost + 1,
                neighbor,
                path + [neighbor]
            ))

    print("[WARN] No path found")
    return []

