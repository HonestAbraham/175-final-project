import random

items = [
    'beef',
    'porkchop',
    # 'fish',
    'rabbit',
    'coal'

]


# food_recipes = {
#         'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
#         'pumpkin_seeds': ['pumpkin'],
#         'bowl': ['planks', 'planks'],
#         'mushroom_stew': ['bowl', 'red_mushroom']
#     }

food_recipes = {}  


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
    'cooked_rabbit': 5
}


def is_solution(reward):
    return reward == 6

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