from __future__ import division

import MalmoPython
import random
import sys
import time
import random
import math
import hunger_learner_helper as submission
from Morgan_agent import Morgan
from a_star import enhance_morgan_with_pathfinding  # Import the pathfinding enhancement
import torch

items=submission.items

def buildPositionList(items):
    """Places the items in a circle."""
    positions = []
    angle = 2*math.pi/len(items)
    for i in range(len(items)):
        x = int(9*math.sin(i*angle))
        y = int(9*math.cos(i*angle))
        positions.append((x, y))
    return positions


def getItemDrawing(positions):
    """Create the XML for the items."""
    drawing = ""
    index = 0
    for p in positions:
        item = items[index].split()
        drawing += '<DrawItem x="' + str(p[0]) + '" y="228" z="' + str(p[1]) + '" type="' + item[0]
        if len(item) > 1:
            drawing += '" variant="' + item[1]
        drawing += '" />'
        index += 1
    return drawing

def generateRandomFurnaceLocation():
    """Generate random furnace location"""
    # Avoid spawn area and make sure it's reachable
    possible_locations = []
    for x in range(-8, 9):
        for z in range(-8, 9):
            # Not too close to spawn
            if abs(x) > 3 or abs(z) > 3:
                possible_locations.append((x, z))
    
    return random.choice(possible_locations)

def generateRandomCraftingTableLocation():
    """Generate random crafting table location"""
    possible_locations = []
    for x in range(-8, 9):
        for z in range(-8, 9):
            # Not too close to spawn
            if abs(x) > 3 or abs(z) > 3:
                possible_locations.append((x, z))
    
    return random.choice(possible_locations)

def generateRandomChestLocation():
    """Generate random chest location"""
    possible_locations = []
    for x in range(-8, 9):
        for z in range(-8, 9):
            # Not too close to spawn
            if abs(x) > 3 or abs(z) > 3:
                possible_locations.append((x, z))
    
    return random.choice(possible_locations)

def generateHouseStructure():
    """Generate a one-story house structure without roof"""
    house_elements = []
    
    # House dimensions (21x21)
    house_size = 10  # This gives us 21x21 (-10 to +10)
    
    # Foundation and floor
    for x in range(-house_size, house_size + 1):
        for z in range(-house_size, house_size + 1):
            house_elements.append((x, 226, z, "planks"))  # Solid wooden floor
    
    # Walls
    for x in range(-house_size, house_size + 1):
        for y in range(227, 231):  # 4 blocks high walls
            # Front and back walls
            house_elements.append((x, y, -house_size, "planks"))
            house_elements.append((x, y, house_size, "planks"))
    
    for z in range(-house_size, house_size + 1):
        for y in range(227, 231):  # 4 blocks high walls
            # Side walls
            house_elements.append((-house_size, y, z, "planks"))
            house_elements.append((house_size, y, z, "planks"))
    
    # Door (front of house)
    house_elements.append((0, 227, -house_size, "air"))  # Door bottom
    house_elements.append((0, 228, -house_size, "air"))  # Door top
    
    # Windows
    window_positions = [
        (-5, 228, -house_size),  # Front windows
        (5, 228, -house_size),
        (-5, 228, house_size),   # Back windows
        (5, 228, house_size),
        (-house_size, 228, -5),  # Side windows
        (-house_size, 228, 5),
        (house_size, 228, -5),
        (house_size, 228, 5)
    ]
    
    for wx, wy, wz in window_positions:
        house_elements.append((wx, wy, wz, "glass"))
    
    return house_elements

def getHouseDrawing(house_elements):
    """Create XML for house structure"""
    drawing = ""
    for x, y, z, block_type in house_elements:
        drawing += f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{block_type}"/>'
    return drawing

def GetMissionXML(summary):
    ''' Build an XML mission string with a house structure and random furnace/crafting table/chest locations. '''

    positions = buildPositionList(items)
    
    # Generate random elements for this mission
    furnace_x, furnace_z = generateRandomFurnaceLocation()
    crafting_x, crafting_z = generateRandomCraftingTableLocation()
    chest_x, chest_z = generateRandomChestLocation()
    house_structure = generateHouseStructure()

    mission_xml = '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>

        <ModSettings>
            <MsPerTick>100</MsPerTick>
        </ModSettings>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <DrawCuboid x1="-50" y1="226" z1="-50" x2="50" y2="236" z2="50" type="air" />
                    <DrawCuboid x1="-50" y1="225" z1="-50" x2="50" y2="225" z2="50" type="bedrock" />
                    ''' + getHouseDrawing(house_structure) + '''
                    ''' + getItemDrawing(positions) + '''
                    <DrawBlock x="''' + str(crafting_x) + '''" y="227" z="''' + str(crafting_z) + '''" type="crafting_table"/>
                    <DrawBlock x="''' + str(furnace_x) + '''" y="227" z="''' + str(furnace_z) + '''" type="furnace"/>
                    <DrawBlock x="''' + str(chest_x) + '''" y="227" z="''' + str(chest_z) + '''" type="chest"/>
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>morgan</Name>
            <AgentStart>
                <Placement x="0.5" y="227.0" z="0.5"/>
                <Inventory>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="480"/>
                <AbsoluteMovementCommands/>
                <ObservationFromFullStats/>
                <SimpleCraftCommands/>
                <MissionQuitCommands/>
                <InventoryCommands/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="40" yrange="40" zrange="40"/>
                </ObservationFromNearbyEntities>
                <ObservationFromGrid>
                    <Grid name="floor_all">
                        <min x="-10" y="0" z="-10"/>
                        <max x="10" y="0" z="10"/>
                    </Grid>
                </ObservationFromGrid>
                <ObservationFromFullInventory/>
                <AgentQuitFromCollectingItem>
                    <Item type="rabbit_stew" description="Supper's Up!!"/>
                </AgentQuitFromCollectingItem>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''

    return mission_xml, None, (furnace_x, furnace_z), (crafting_x, crafting_z), (chest_x, chest_z)

def validate_learning_parameters(morgan):
    """Validate learning parameters and provide warnings if needed"""
    warnings = []
    
    if morgan.use_dqn:
        # Check DQN learning rate
        if hasattr(morgan, 'optimizer'):
            lr = morgan.optimizer.param_groups[0]['lr']
            if lr > 0.01:
                warnings.append(f"WARNING: DQN learning rate ({lr}) is too high. Recommended range: 0.0001-0.001")
            elif lr < 0.0001:
                warnings.append(f"WARNING: DQN learning rate ({lr}) is very low. This might slow down learning.")
    else:
        # Check Q-learning rate
        if morgan.n > 0.5:
            warnings.append(f"WARNING: Q-learning rate ({morgan.n}) is high. This might cause unstable learning.")
        elif morgan.n < 0.1:
            warnings.append(f"WARNING: Q-learning rate ({morgan.n}) is low. This might slow down learning.")
    
    return warnings

def display_agent_statistics(morgan, total_episodes, successful_episodes, avg_completion_times):
    """Display comprehensive statistics about the agent's performance."""
    print("\n=== AGENT TRAINING STATISTICS ===")
    
    # Check for learning parameter warnings
    warnings = validate_learning_parameters(morgan)
    if warnings:
        print("\n=== LEARNING PARAMETER WARNINGS ===")
        for warning in warnings:
            print(warning)
    
    print(f"Total Episodes Trained: {total_episodes}")
    print(f"Successful Episodes: {successful_episodes}")
    success_rate = (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0
    print(f"Success Rate: {success_rate:.2f}%")
    
    if total_episodes > 0:
        avg_time = sum(avg_completion_times) / len(avg_completion_times) if avg_completion_times else 0
        print(f"\nTime Statistics:")
        print(f"Average Completion Time: {avg_time:.2f} seconds")
        print(f"Best Completion Time: {min(avg_completion_times) if avg_completion_times else 0:.2f} seconds")
        
        # Calculate episodes to first success
        if successful_episodes > 0:
            first_success_episode = total_episodes - len(avg_completion_times) + 1
            print(f"\nLearning Metrics:")
            print(f"Episodes to First Success: {first_success_episode}")
            print(f"Learning Rate: {success_rate / first_success_episode:.2f}% per episode")
    
    print("\nLearning Parameters:")
    # Safely access attributes that might not exist
    n_value = getattr(morgan, 'n', 'N/A')
    dqn_value = getattr(morgan, 'use_dqn', 'N/A')
    print(f"Learning Rate (n): {n_value}")
    print(f"Using DQN: {dqn_value}")
    
    # Display learning rate analysis
    if not morgan.use_dqn:
        lr_analysis = morgan.get_learning_rate_analysis()
        if lr_analysis:
            print("\nLearning Rate Analysis (Last 30 Updates):")
            print(f"Current Learning Rate: {lr_analysis['learning_rate']}")
            print(f"Average Relative Change: {lr_analysis['avg_relative_change']:.4f}")
            print(f"Average Absolute Change: {lr_analysis['avg_absolute_change']:.4f}")
            print("\nLearning Rate Effects:")
            print("- Higher learning rate (>0.5): Faster learning but more unstable")
            print("- Lower learning rate (<0.1): More stable but slower learning")
            print("- Current setting: " + 
                  ("High - Fast learning, might be unstable" if n_value > 0.5 else
                   "Low - Stable but might learn slowly" if n_value < 0.1 else
                   "Moderate - Balanced learning speed and stability"))
    
    # Display top 5 most valuable state-action pairs if available
    if hasattr(morgan, 'q_table') and morgan.q_table:
        print("\nTop 5 Most Valuable State-Action Pairs:")
        q_values = [(state_action, value) for state_action, value in morgan.q_table.items()]
        q_values.sort(key=lambda x: x[1], reverse=True)
        for i, (state_action, value) in enumerate(q_values[:5]):
            print(f"{i+1}. State-Action: {state_action}, Value: {value:.2f}")
        
        # Show Q-value change metrics for tabular Q-learning
        if not morgan.use_dqn:
            q_metrics = morgan.get_q_learning_metrics()
            if q_metrics:
                print("\nQ-Learning Performance Metrics (Last 30 Episodes):")
                print(f"Average Q-Value Change: {q_metrics['avg_q_change']:.4f}")
                print(f"Q-Value Change Std Dev: {q_metrics['q_change_std_dev']:.4f}")
                print(f"Recent Q-Value Changes: {[f'{x:.4f}' for x in q_metrics['recent_changes']]}")
                print(f"Reward Std Dev: {q_metrics['reward_std_dev']:.4f}")
                if q_metrics['recent_rewards']:
                    print(f"Recent Rewards: {[f'{x:.2f}' for x in q_metrics['recent_rewards']]}")
                print(f"Trend: {'Decreasing' if q_metrics['recent_changes'][-1] < q_metrics['recent_changes'][0] else 'Increasing'}")
    
    # For DQN, show additional metrics
    if dqn_value:
        dqn_metrics = morgan.get_dqn_metrics()
        if dqn_metrics:
            print("\nDQN Performance Metrics (Last 30 Episodes):")
            print(f"Average Recent Reward: {dqn_metrics['avg_recent_reward']:.2f}")
            print(f"Reward Std Dev: {dqn_metrics['reward_std_dev']:.4f}")
            print(f"Average Recent Loss: {dqn_metrics['avg_recent_loss']:.4f}")
            print(f"Loss Std Dev: {dqn_metrics['loss_std_dev']:.4f}")
            
            if dqn_metrics['recent_rewards']:
                print(f"Recent Rewards: {[f'{x:.2f}' for x in dqn_metrics['recent_rewards']]}")
            if dqn_metrics['recent_losses']:
                print(f"Recent Losses: {[f'{x:.4f}' for x in dqn_metrics['recent_losses']]}")
            
            if hasattr(morgan, 'loss_history') and len(morgan.loss_history) >= 2:
                print(f"Loss Trend: {'Decreasing' if morgan.loss_history[-1] < morgan.loss_history[0] else 'Increasing'}")
                
                # Display loss history in chunks
                print("\nLoss History (by episode):")
                chunk_size = 10
                for i in range(0, len(morgan.loss_history), chunk_size):
                    chunk = morgan.loss_history[i:i+chunk_size]
                    episodes = range(i+1, i+len(chunk)+1)
                    print(f"Episodes {episodes[0]}-{episodes[-1]}: {[f'{x:.4f}' for x in chunk]}")

def save_training_data(morgan, total_episodes, successful_episodes, avg_completion_times, first_success_episode, recent_successes, episode_rewards, filename="training_data.txt"):
    """Save training data to file"""
    with open(filename, 'w') as f:
        f.write("=== Training Statistics ===\n")
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"Successful Episodes: {successful_episodes}\n")
        f.write(f"Success Rate: {(successful_episodes/total_episodes*100):.2f}%\n")
        f.write(f"First Success at Episode: {first_success_episode if first_success_episode else 'Not yet'}\n")
        
        if avg_completion_times:
            f.write(f"\nCompletion Times:\n")
            f.write(f"Average: {sum(avg_completion_times)/len(avg_completion_times):.2f} seconds\n")
            f.write(f"Best: {min(avg_completion_times):.2f} seconds\n")
        
        if recent_successes:
            f.write(f"\nRecent Success Rate (last {len(recent_successes)}): {(sum(recent_successes)/len(recent_successes)*100):.1f}%\n")
        
        if episode_rewards:
            f.write(f"\nReward Statistics:\n")
            f.write(f"Average Reward (last 10): {sum(episode_rewards[-10:])/min(10, len(episode_rewards)):.2f}\n")
            f.write(f"Best Reward: {max(episode_rewards):.2f}\n")
        
        f.write(f"\nLearning Parameters:\n")
        f.write(f"Learning Rate (n): {getattr(morgan, 'n', 'N/A')}\n")
        f.write(f"Using DQN: {getattr(morgan, 'use_dqn', 'N/A')}\n")
        
        # Save loss history if available
        if hasattr(morgan, 'loss_history'):
            f.write("\nLoss History:\n")
            for i, loss in enumerate(morgan.loss_history):
                f.write(f"Episode {i+1}: {loss:.4f}\n")
        
        # Save Q-value changes for tabular Q-learning
        if not morgan.use_dqn and hasattr(morgan, 'q_value_changes'):
            f.write("\nQ-Value Changes History:\n")
            for i, change in enumerate(morgan.q_value_changes):
                f.write(f"Episode {i+1}: {change:.4f}\n")
        
        # Save Q-table or DQN state if available
        if hasattr(morgan, 'q_table') and morgan.q_table:
            f.write("\nQ-Table State:\n")
            for state, actions in morgan.q_table.items():
                f.write(f"State: {state}\n")
                for action, value in actions.items():
                    f.write(f"  Action: {action.ljust(20)} Q-value: {value:.2f}\n")
                f.write("\n")

def analyze_best_strategies(morgan):
    """Analyze and display the best strategies the model has learned"""
    print("\n=== Best Learned Strategies ===")
    
    if morgan.use_dqn:
        print("\nDQN Strategy Analysis:")
        # Get the best states and actions from DQN
        if hasattr(morgan, 'episode_rewards') and morgan.episode_rewards:
            best_episode_idx = morgan.episode_rewards.index(max(morgan.episode_rewards))
            print(f"Best Episode Reward: {morgan.episode_rewards[best_episode_idx]:.2f}")
            
            # Analyze the policy network's predictions
            print("\nTop 5 Recommended Actions:")
            # Create a sample state with one of each item
            sample_state = tuple((item, 1) for item in morgan.item_list)
            state_tensor = morgan.state_to_tensor(sample_state).unsqueeze(0)
            
            with torch.no_grad():
                q_values = morgan.policy_net(state_tensor)
                values, indices = torch.topk(q_values, 5)
                
                for i, (value, idx) in enumerate(zip(values[0], indices[0])):
                    action = morgan.all_actions[idx.item()]
                    print(f"{i+1}. Action: {action}")
                    print(f"   Expected Value: {value.item():.2f}")
                    
                    # Show what this action produces
                    if action in submission.food_recipes:
                        print(f"   Creates: {action} using {submission.food_recipes[action]}")
                    elif action in submission.cooking_recipes:
                        print(f"   Creates: {action} using {submission.cooking_recipes[action]}")
    else:
        print("\nQ-Learning Strategy Analysis:")
        if hasattr(morgan, 'q_table') and morgan.q_table:
            # Find states with highest Q-values
            best_states = []
            for state, actions in morgan.q_table.items():
                if not actions:
                    continue
                best_action = max(actions.items(), key=lambda x: x[1])
                best_states.append((state, best_action[0], best_action[1]))
            
            # Sort by Q-value
            best_states.sort(key=lambda x: x[2], reverse=True)
            
            print("\nTop 5 Recommended Strategies:")
            for i, (state, action, value) in enumerate(best_states[:5]):
                print(f"\nStrategy {i+1}:")
                print(f"Current Items: {dict(state)}")
                print(f"Recommended Action: {action}")
                print(f"Expected Value: {value:.2f}")
                
                # Show what this action produces
                if action in submission.food_recipes:
                    print(f"Creates: {action} using {submission.food_recipes[action]}")
                elif action in submission.cooking_recipes:
                    print(f"Creates: {action} using {submission.cooking_recipes[action]}")

def save_loss_data_for_plotting(morgan, filename="loss_data.csv"):
    """Save loss data in CSV format for easy plotting"""
    if not hasattr(morgan, 'loss_history') or not morgan.loss_history:
        print("No loss history available")
        return
        
    # Get the last episode number from the file if it exists
    last_episode = 0
    try:
        with open(filename, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                episode = int(line.split(',')[0])
                last_episode = max(last_episode, episode)
    except FileNotFoundError:
        pass
    
    # Append new data starting from the last episode
    with open(filename, 'a') as f:
        # Write header only if file is new
        if last_episode == 0:
            f.write("Episode,Loss\n")
        
        # Write new data
        for i, loss in enumerate(morgan.loss_history, 1):
            f.write(f"{last_episode + i},{loss}\n")
    
    print(f"Loss data saved to {filename} (Total episodes: {last_episode + len(morgan.loss_history)})")

if __name__ == '__main__':
    random.seed(2)
    
    expected_reward = 3390
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    # Get number of episodes from command line argument or use default
    num_reps = 30  # Default to 30 episodes
    if len(sys.argv) > 1:
        try:
            num_reps = int(sys.argv[1])
        except ValueError:
            print("Invalid number of episodes. Using default of 30.")
    
    n=1
    
    # Create original Morgan agent
    morgan = Morgan(n=n, use_dqn=True)
    
    # # Reset training state to start fresh
    # morgan.reset_training_state()
    
    # Initialize pathfinder with known obstacle positions
    house_structure = generateHouseStructure()
    obstacles = [(x, z, block_type) for x, y, z, block_type in house_structure]
    morgan.pathfinder.set_known_obstacles(obstacles)
    
    # Statistics tracking
    successful_episodes = 0
    avg_completion_times = []
    episode_start_time = None
    first_success_episode = None
    recent_successes = []  # Track last 10 episodes
    episode_rewards = []  # Track rewards for all episodes
    
    print(f"\nStarting training for {num_reps} episodes...")
    
    try:
        for iRepeat in range(num_reps):
            print(f"\n=== EPISODE {iRepeat + 1}/{num_reps} ===")
            episode_start_time = time.time()
            
            # Generate mission with random locations
            mission_xml, _, furnace_pos, crafting_pos, chest_pos = GetMissionXML("Food Search #" + str(iRepeat))
            
            # Update pathfinder with new furnace and crafting table positions
            morgan.pathfinder.set_known_obstacles(obstacles, furnace_pos=furnace_pos, crafting_pos=crafting_pos)
            
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission_record = MalmoPython.MissionRecordSpec()
            my_mission.requestVideo(800, 500)
            my_mission.setViewpoint(1)
            max_retries = 3

            # Clear inventory for Morgan
            morgan.clear_inventory()

            for retry in range(max_retries):
                try:
                    agent_host.startMission(my_mission, my_client_pool, my_mission_record, 0, "morgan")
                    break
                except RuntimeError as e:
                    if retry == max_retries - 1:
                        print("Error starting mission:", e)
                        print("Is the game running?")
                        exit(1)
                    else:
                        time.sleep(0.5)

            world_state = agent_host.getWorldState()
            while not world_state.has_mission_begun:
                time.sleep(0.05)
                world_state = agent_host.getWorldState()

            # Run the episode
            success = morgan.run(agent_host)
            episode_time = time.time() - episode_start_time
            
            # Track episode results
            if success:
                successful_episodes += 1
                avg_completion_times.append(episode_time)
                if first_success_episode is None:
                    first_success_episode = iRepeat + 1
            
            # Track recent success rate (last 10 episodes)
            recent_successes.append(1 if success else 0)
            if len(recent_successes) > 10:
                recent_successes.pop(0)
            
            # Track episode reward
            if hasattr(morgan, 'episode_rewards') and morgan.episode_rewards:
                episode_rewards.append(morgan.episode_rewards[-1])
            
            # Display progress every 10 episodes or at the end
            if (iRepeat + 1) % 10 == 0 or iRepeat == num_reps - 1:
                print("\n=== Training Progress ===")
                print(f"Episodes completed: {iRepeat + 1}/{num_reps}")
                print(f"Total successful episodes: {successful_episodes}")
                print(f"First success at episode: {first_success_episode if first_success_episode else 'Not yet'}")
                print(f"Recent success rate (last 10): {(sum(recent_successes) / len(recent_successes)) * 100:.1f}%")
                if episode_rewards:
                    print(f"Average reward (last 10): {sum(episode_rewards[-10:]) / min(10, len(episode_rewards)):.2f}")
                
                # Save data
                save_training_data(morgan, iRepeat + 1, successful_episodes, avg_completion_times, 
                                 first_success_episode, recent_successes, episode_rewards)
            
            # Clear inventory after each episode
            morgan.clear_inventory()
            # Add a delay between missions to ensure proper cleanup
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    except Exception as e:
        print(f"\nTraining interrupted by error: {e}")
    finally:
        print("\nSaving final training data...")
        save_training_data(morgan, iRepeat + 1, successful_episodes, avg_completion_times, 
                         first_success_episode, recent_successes, episode_rewards)
        print("Training data saved to training_data.txt")
        
        # Save loss data for plotting
        save_loss_data_for_plotting(morgan)
        
        print("\nFinal Statistics:")
        display_agent_statistics(morgan, iRepeat + 1, successful_episodes, avg_completion_times)