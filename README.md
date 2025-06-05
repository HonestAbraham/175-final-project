# 175-final-project

Cooked Food:
- Bread (minecraft:bread)
- Baked Potato (minecraft:baked_potato)
- Cooked Beef / Steak (minecraft:cooked_beef)
- Cooked Chicken (minecraft:cooked_chicken)
- Cooked Fish (minecraft:cooked_fish)
- Cooked Salmon (minecraft:cooked_fish, DataValue 1)
- Cooked Mutton (minecraft:cooked_mutton)
- Cooked Porkchop (minecraft:cooked_porkchop)
- Cooked Rabbit (minecraft:cooked_rabbit)
- Rabbit Stew (minecraft:rabbit_stew)
- Beetroot Soup (minecraft:beetroot_soup)
- Mushroom Stew (minecraft:mushroom_stew)
- Pumpkin Pie (minecraft:pumpkin_pie)

Raw Ingredients / Raw Food:
- Apple (minecraft:apple)
- Raw Beef (minecraft:beef)
- Raw Chicken (minecraft:chicken)
- Raw Fish (minecraft:fish)
- Raw Salmon (minecraft:fish, DataValue 1)
- Clownfish (Tropical Fish) (minecraft:fish, DataValue 2)
- Pufferfish (minecraft:fish, DataValue 3)
- Raw Mutton (minecraft:mutton)
- Raw Porkchop (minecraft:porkchop)
- Raw Rabbit (minecraft:rabbit)
- Potato (minecraft:potato)
- Poisonous Potato (minecraft:poisonous_potato)
- Beetroot (minecraft:beetroot)
- Carrot (minecraft:carrot)
- Melon Slice (minecraft:melon)
- Pumpkin (minecraft:pumpkin)
- Egg (minecraft:egg)
- Sugar (minecraft:sugar)
- Milk Bucket (minecraft:milk_bucket)
- Wheat (minecraft:wheat)
- Cookie (minecraft:cookie)
- Glistering Melon Slice (minecraft:speckled_melon)
- Spider Eye (minecraft:spider_eye)

TODO:
- agent freezes (not sure what to do about that)
- agent can only walk instead of run
- increase scope of food (last)
    - update food recipes and cook recipes
a. use furnace to cook and crafting table to craft 
    - navigate to furnace & crafting table dynamically
- make it walk from one ingredient to another rather than teleport back to the middle
    - try to avoid extra ingredients or drop them
    - pathfind
- maybe craft the crafting table and furnace?
- instead of using predetermined rewards map, use the hunger gained as reward.
    - max reward/goal reward is incorrect. we need it to only be able to consume one food at a time.
    - only be able to consume when hunger is not max
        - even when hunger is not max, sometimes hunger gained isnt always what u get, it might be capped at 10.
            - for example: current hunger is 8, eating a steak will result in 10 hunger, giving a reward of 2. But reward should actually be 8

install torch and torchvision


CHAI TO DO A*:
Missing Maze Awareness 
The A* algorithm needs to know about the maze obstacles to plan around them. Your current setup generates random obstacles, but there's no clear indication that the A* algorithm receives this obstacle information.

Stuck Prevention Not Implemented 
A* pathfinding can get stuck when agents try to follow paths without considering dynamic obstacles or when they encounter situations where the calculated path becomes invalid

the agent itself is not taking in the obstacle properly
the a* stuck prevention is weird as fuck
make sure the maze is proper, like a path no random blocks\
the agent still assumes that it reach the furnace by force move (FLAWED STUCK LOGIC)

test case:
box the agent with a wall
make sure the agent reach the furnace to cook and dont skip it



worst case:
fuck the maze just put a the crafting table and furnace randomly already works