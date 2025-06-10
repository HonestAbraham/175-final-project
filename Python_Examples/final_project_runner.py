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


if __name__ == '__main__':
    random.seed(0)
    
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
    morgan = Morgan(n=n, use_dqn=True)
    
    # Initialize pathfinder with known obstacle positions
    house_structure = generateHouseStructure()
    obstacles = [(x, z, block_type) for x, y, z, block_type in house_structure]
    morgan.pathfinder.set_known_obstacles(obstacles)
    
    for iRepeat in range(num_reps):
        print(f"\n=== EPISODE {iRepeat + 1} ===")
        
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
                    time.sleep(2)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

        # Every few iteration morgan will show us the best policy that he learned.
        if (iRepeat + 1) % 5 == 0:
            found_solution = morgan.best_policy(agent_host)
            if found_solution:
                print('Found solution!')
                break
            time.sleep(2)
            world_state = agent_host.getWorldState()
            if world_state.is_mission_running:
                agent_host.sendCommand('quit')
                time.sleep(1)
        else:
            morgan.run(agent_host)

        # Clear inventory after each episode
        morgan.clear_inventory()
        # Add a delay between missions to ensure proper cleanup
        time.sleep(2)
        
    print("Training completed!")