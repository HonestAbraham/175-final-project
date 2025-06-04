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

items=submission.items

def buildPositionList(items):
    """Places the items in a circle."""
    positions = []
    angle = 2*math.pi/len(items)
    for i in range(len(items)):
        x = int(6*math.sin(i*angle))
        y = int(6*math.cos(i*angle))
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


def generateMazeWalls():
    """Generate maze-like wall structures"""
    walls = []
    
    # Create L-shaped corridors
    # Vertical wall 1
    for z in range(-7, -2):
        walls.append((-6, z, "stone"))
    
    # Horizontal wall 1
    for x in range(-6, -1):
        walls.append((x, -2, "cobblestone"))
    
    # Vertical wall 2 (creates a corridor)
    for z in range(2, 7):
        walls.append((4, z, "stone"))
    
    # Horizontal wall 2
    for x in range(1, 5):
        walls.append((x, 2, "cobblestone"))
    
    # Create some scattered obstacles
    scattered_obstacles = [
        (-3, 5, "log"),
        (2, -5, "log"),
        (-8, 1, "stone"),
        (7, -3, "cobblestone"),
        (-1, 7, "water"),
        (6, 6, "water")
    ]
    walls.extend(scattered_obstacles)
    
    return walls

def generateComplexMaze():
    """Generate a more complex maze structure"""
    maze = []
    
    # Create a spiral-like maze
    # Outer walls
    for x in range(-8, 9):
        if x not in [-1, 0, 1]:  # Leave gaps
            maze.append((x, -8, "stone"))
            maze.append((x, 8, "stone"))
    
    for z in range(-8, 9):
        if z not in [-1, 0, 1]:  # Leave gaps
            maze.append((-8, z, "stone"))
            maze.append((8, z, "stone"))
    
    # Inner maze walls creating corridors
    # Horizontal corridors
    for x in range(-6, 7):
        if x not in [-2, -1, 0, 1, 2]:  # Gaps for passage
            maze.append((x, -4, "cobblestone"))
            maze.append((x, 4, "cobblestone"))
    
    # Vertical corridors
    for z in range(-6, 7):
        if z not in [-2, -1, 0, 1, 2]:  # Gaps for passage
            maze.append((-4, z, "cobblestone"))
            maze.append((4, z, "cobblestone"))
    
    # Add some water obstacles for variety
    water_spots = [
        (-6, -6, "water"), (6, -6, "water"),
        (-6, 6, "water"), (6, 6, "water"),
        (-2, -7, "water"), (2, 7, "water")
    ]
    maze.extend(water_spots)
    
    return maze

def generateRandomMaze():
    """Generate a random maze using maze generation algorithm"""
    maze = []
    
    # Create a grid-based maze
    maze_size = 15  # -7 to 7
    
    # Create walls in a checkerboard pattern with random gaps
    for x in range(-7, 8, 2):
        for z in range(-7, 8, 2):
            # Skip spawn area
            if abs(x) <= 2 and abs(z) <= 2:
                continue
            
            # Randomly place walls
            if random.random() < 0.7:  # 70% chance of wall
                maze.append((x, z, random.choice(["stone", "cobblestone"])))
            
            # Create connecting walls (sometimes)
            if random.random() < 0.4 and x < 6:  # 40% chance of horizontal connection
                maze.append((x + 1, z, random.choice(["stone", "cobblestone"])))
            
            if random.random() < 0.4 and z < 6:  # 40% chance of vertical connection
                maze.append((x, z + 1, random.choice(["stone", "cobblestone"])))
    
    # Add some water hazards
    for _ in range(3):
        x = random.randint(-6, 6)
        z = random.randint(-6, 6)
        if abs(x) > 2 or abs(z) > 2:  # Not in spawn area
            maze.append((x, z, "water"))
    
    return maze

def generateRandomObstacles():
    """Generate different types of obstacle patterns"""
    obstacle_type = random.choice(["simple", "maze", "complex", "random_maze"])
    
    if obstacle_type == "simple":
        # Original simple random obstacles
        obstacles = []
        num_obstacles = random.randint(8, 15)
        
        for _ in range(num_obstacles):
            x = random.randint(-9, 9)
            z = random.randint(-9, 9)
            
            if abs(x) <= 2 and abs(z) <= 2:
                continue
                
            obstacle_block = random.choice(["stone", "cobblestone", "log", "water"])
            obstacles.append((x, z, obstacle_block))
        
        print(f"[MAZE TYPE] Simple random obstacles")
        return obstacles
    
    elif obstacle_type == "maze":
        print(f"[MAZE TYPE] L-shaped corridor maze")
        return generateMazeWalls()
    
    elif obstacle_type == "complex":
        print(f"[MAZE TYPE] Complex spiral maze")
        return generateComplexMaze()
    
    elif obstacle_type == "random_maze":
        print(f"[MAZE TYPE] Random grid-based maze")
        return generateRandomMaze()
    
    return []

def generateRandomFurnaceLocation():
    """Generate random furnace location to test pathfinding"""
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

def getObstacleDrawing(obstacles):
    """Create XML for random obstacles"""
    drawing = ""
    for x, z, obstacle_type in obstacles:
        drawing += f'<DrawBlock x="{x}" y="227" z="{z}" type="{obstacle_type}"/>'
    return drawing

def GetMissionXML(summary):
    ''' Build an XML mission string with random obstacles and furnace location. Returns XML and obstacle data.'''

    positions = buildPositionList(items)
    
    # Generate random elements for this mission
    obstacles = generateRandomObstacles()
    furnace_x, furnace_z = generateRandomFurnaceLocation()
    crafting_x, crafting_z = generateRandomCraftingTableLocation()
    
    print(f"[MISSION SETUP] Furnace at ({furnace_x}, {furnace_z})")
    print(f"[MISSION SETUP] Crafting table at ({crafting_x}, {crafting_z})")
    print(f"[MISSION SETUP] {len(obstacles)} obstacles placed")

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
            </ServerSection>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <DrawCuboid x1="-50" y1="226" z1="-50" x2="50" y2="228" z2="50" type="air" />
                    <DrawCuboid x1="-50" y1="226" z1="-50" x2="50" y2="226" z2="50" type="monster_egg" variant="chiseled_brick" />
                    <DrawCuboid x1="-3" y1="226" z1="-3" x2="3" y2="226" z2="3" type="dirt" />
                    <DrawBlock x="-0" y="226" z="0" type="diamond_block"/>
                    ''' + getItemDrawing(positions) + '''
                    <DrawBlock x="''' + str(crafting_x) + '''" y="227" z="''' + str(crafting_z) + '''" type="crafting_table"/>
                    <DrawBlock x="''' + str(furnace_x) + '''" y="227" z="''' + str(furnace_z) + '''" type="furnace"/>
                    ''' + getObstacleDrawing(obstacles) + '''
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

    return mission_xml, obstacles, (furnace_x, furnace_z), (crafting_x, crafting_z)


if __name__ == '__main__':
    random.seed(0)
    print('Starting with A* Pathfinding and Random Obstacles...', flush=True)

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

    num_reps = 30000
    n=1
    
    # Create original Morgan agent
    morgan = Morgan(n=n)
    print("n=",n)
    print("Enhancing Morgan with A* pathfinding...")
    
    for iRepeat in range(num_reps):
        print(f"\n=== EPISODE {iRepeat + 1} ===")
        
        # Generate mission with obstacle data
        mission_xml, obstacles, furnace_pos, crafting_pos = GetMissionXML("Food Search with A* #" + str(iRepeat))
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)
        max_retries = 3

        # Enhance Morgan with A* pathfinding capabilities
        enhanced_morgan = enhance_morgan_with_pathfinding(morgan)
        
        # Set known obstacles BEFORE starting mission
        enhanced_morgan.pathfinder.set_known_obstacles(obstacles, furnace_pos, crafting_pos)
        print("A* pathfinding enhancement complete with obstacle awareness!")
        
        # Clear inventory for the enhanced Morgan
        enhanced_morgan.morgan.clear_inventory()

        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_client_pool, my_mission_record, 0, "morgan")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission", e)
                    print("Is the game running?")
                    exit(1)
                else:
                    time.sleep(2)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

        # Every few iteration morgan will show us the best policy that he learned.
        if (iRepeat + 1) % 5 == 0:
            print((iRepeat+1), 'Showing best policy with A*:', end = " ")
            found_solution = enhanced_morgan.morgan.best_policy(agent_host)
            if found_solution:
                print('Found solution with A* pathfinding!')
                print('Done')
                break
        else:
            print((iRepeat+1), 'Learning Q-Table with A*:', end = " ")
            enhanced_morgan.morgan.run(agent_host)

        # Clear inventory after each episode
        enhanced_morgan.morgan.clear_inventory()
        time.sleep(1)
        
    print("Training completed with A* pathfinding integration!")