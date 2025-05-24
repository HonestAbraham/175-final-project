
from __future__ import division

import MalmoPython
import random
import sys
import time
import random
import math
import hunger_learner_helper as submission
from Morgan_agent import Morgan

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


def GetMissionXML(summary):
    ''' Build an XML mission string that uses the RewardForCollectingItem mission handler.'''

    positions = buildPositionList(items)

    return '''<?xml version="1.0" encoding="UTF-8" ?>
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
                    <DrawCuboid x1="-50" y1="226" z1="-50" x2="50" y2="228" z2="50" type="air" />
                    <DrawCuboid x1="-50" y1="226" z1="-50" x2="50" y2="226" z2="50" type="monster_egg" variant="chiseled_brick" />
                    <DrawCuboid x1="-3" y1="226" z1="-3" x2="3" y2="226" z2="3" type="dirt" />
                    <DrawBlock x="-0" y="226" z="0" type="diamond_block"/>
                    ''' + getItemDrawing(positions) + '''
                    <DrawBlock x="5" y="227" z="5" type="crafting_table"/>
                    <DrawBlock x="-5" y="227" z="-5" type="furnace"/>
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


if __name__ == '__main__':
    random.seed(0)
    #sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
    print('Starting...', flush=True)

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
    morgan = Morgan(n=n)
    print("n=",n)
    morgan.clear_inventory()
    for iRepeat in range(num_reps):
        my_mission = MalmoPython.MissionSpec(GetMissionXML("Food Search #" + str(iRepeat)), True)
        my_mission_record = MalmoPython.MissionRecordSpec()  # Records nothing by default
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Attempt to start the mission:
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
            print((iRepeat+1), 'Showing best policy:', end = " ")
            found_solution = morgan.best_policy(agent_host)
            if found_solution:
                print('Found solution')
                print('Done')
                break
        else:
            print((iRepeat+1), 'Learning Q-Table:', end = " ")
            morgan.run(agent_host)

        morgan.clear_inventory()
        time.sleep(1)
