import heapq
import math
import json
import time
from typing import List, Tuple, Set, Optional, Dict
from collections import deque
import sys

# Import Morgan and related modules from your existing file
from Morgan_agent import Morgan
import hunger_learner_helper as submission

class Node:
    def __init__(self, x: int, z: int, g_cost: float = 0, h_cost: float = 0, parent=None):
        self.x = x
        self.z = z
        self.g_cost = g_cost  # Cost from start to this node
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.z == other.z
    
    def __hash__(self):
        return hash((self.x, self.z))

class MorganPathfinder:
    def __init__(self, grid_size: int = 21, grid_min: int = -10):
        """
        Initialize pathfinder for Morgan's world
        Args:
            grid_size: Size of the grid (21x21 for -10 to 10 range)
            grid_min: Minimum coordinate value (-10)
        """
        self.grid_size = grid_size
        self.grid_min = grid_min
        self.grid_max = grid_min + grid_size - 1
        self.obstacles = set()  # Set of (x, z) coordinates that are blocked
        self.floor_grid = None  # Cache for the floor grid
        
    def update_world_state(self, agent_host, max_wait_seconds=2):
        """
        Update pathfinder with current world state from Malmo observations
        Integrates with Morgan's existing get_obj_locations method
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                try:
                    obs = json.loads(world_state.observations[-1].text)
                    if 'floor_all' in obs:
                        self.floor_grid = obs['floor_all']
                        self._update_obstacles()
                        # print(f"[PATHFINDER] Updated obstacles from world state: {len(self.obstacles)} obstacles found")
                        return True
                except Exception as e:
                    print(f"Error parsing observation: {e}")
            time.sleep(0.1)
        
        print(f"[WARN] Could not update world state within {max_wait_seconds} seconds")
        return False
    
    def _update_obstacles(self):
        """Update obstacles based on floor grid data"""
        self.obstacles.clear()
        if not self.floor_grid:
            return
            
        # Mark non-walkable blocks as obstacles
        walkable_blocks = ['air', 'grass', 'dirt', 'stone', 'cobblestone']
        
        for idx, block in enumerate(self.floor_grid):
            if block not in walkable_blocks:
                x_idx = idx % self.grid_size
                z_idx = idx // self.grid_size
                
                world_x = self.grid_min + x_idx
                world_z = self.grid_min + z_idx
                
                # Don't mark crafting tables and furnaces as obstacles
                if block not in ['crafting_table', 'furnace', 'lit_furnace']:
                    self.obstacles.add((world_x, world_z))
    
    def is_valid_position(self, x: int, z: int, target_pos=None) -> bool:
        """
        Check if position is within bounds and not blocked
        Args:
            x, z: Position to check
            target_pos: If provided, this is our destination - allow movement to furnace/crafting table if this is our target
        """
        # Basic bounds check
        if not (self.grid_min <= x <= self.grid_max and self.grid_min <= z <= self.grid_max):
            return False
            
        pos = (x, z)
        
        # If this is our target position, it's valid even if it's a furnace or crafting table
        if target_pos and pos == target_pos:
            return True
            
        # Check if it's an obstacle
        is_blocked = pos in self.obstacles
        
        # Log obstacle detection
        if is_blocked:
            # print(f"[A*] Position {pos} is blocked by obstacle")
            pass
            
        return not is_blocked
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, node: Node) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-directional movement for reliable navigation)"""
        neighbors = []
        # 4-directional movement: N, E, S, W (no diagonals to avoid wall clipping)
        directions = [
            (0, 1),   # North
            (1, 0),   # East
            (0, -1),  # South
            (-1, 0)   # West
        ]
        
        for dx, dz in directions:
            new_x, new_z = node.x + dx, node.z + dz
            if self.is_valid_position(new_x, new_z):
                neighbors.append((new_x, new_z))
        
        return neighbors
    
    def get_movement_cost(self, current: Tuple[int, int], neighbor: Tuple[int, int]) -> float:
        """Calculate movement cost - uniform cost for 4-directional movement"""
        return 1.0
    
    def find_block_positions(self, agent_host, block_type: str) -> List[Tuple[int, int]]:
        """
        Find all positions of a specific block type
        Integrates with Morgan's existing get_block_position method
        """
        positions = []
        if not self.floor_grid:
            if not self.update_world_state(agent_host):
                return positions
        
        # Search for block in the grid
        for idx, block in enumerate(self.floor_grid):
            if block == block_type or block == f"lit_{block_type}":
                x_idx = idx % self.grid_size
                z_idx = idx // self.grid_size
                
                world_x = self.grid_min + x_idx
                world_z = self.grid_min + z_idx
                positions.append((world_x, world_z))
        
        return positions
    
    def find_nearest_block(self, agent_host, start_pos: Tuple[int, int], block_type: str) -> Optional[Tuple[int, int]]:
        """Find the nearest block of a specific type"""
        block_positions = self.find_block_positions(agent_host, block_type)
        if not block_positions:
            return None
        
        # Find closest block using Manhattan distance
        return min(block_positions, key=lambda pos: self.manhattan_distance(start_pos, pos))
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm with 4-directional movement for reliable navigation
        Returns: List of (x, z) coordinates representing the path, or None if no path found
        """
        # print(f"\n[A*] Starting pathfinding from {start} to {goal}")
        
        # Convert to integers and validate
        start = (int(round(start[0])), int(round(start[1])))
        goal = (int(round(goal[0])), int(round(goal[1])))
        
        # Special handling: if goal is furnace/crafting table, temporarily remove it from obstacles
        is_special_target = goal in self.furnace_positions or goal in self.crafting_positions
        if is_special_target:
            # print(f"[A*] Goal is a special target (furnace/crafting table)")
            self.obstacles.discard(goal)
        
        if not self.is_valid_position(start[0], start[1], goal):
            # print(f"[A*] Invalid start position {start}")
            return None
            
        if not self.is_valid_position(goal[0], goal[1], goal):
            # print(f"[A*] Invalid goal position {goal}")
            return None
        
        if start == goal:
            # print("[A*] Start and goal are the same")
            return [start]
        
        # print(f"[A*] Number of obstacles: {len(self.obstacles)}")
        # print(f"[A*] Obstacles near path: {[obs for obs in self.obstacles if self.manhattan_distance(obs, start) <= 5 or self.manhattan_distance(obs, goal) <= 5]}")
        
        open_set = []
        closed_set = set()
        came_from = {}
        
        # Use Manhattan distance for 4-directional movement
        start_node = Node(start[0], start[1], 0, self.manhattan_distance(start, goal))
        heapq.heappush(open_set, start_node)
        g_costs = {start: 0}
        
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-directional movement
        iterations = 0
        max_iterations = 1000  # Prevent infinite loops
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.z)
            
            if current_pos == goal:
                # print(f"[A*] Path found after {iterations} iterations")
                path = self._reconstruct_path(came_from, current_pos)
                # print(f"[A*] Path length: {len(path)}")
                return path
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            # Try each possible move
            for dx, dz in moves:
                next_x, next_z = current.x + dx, current.z + dz
                next_pos = (next_x, next_z)
                
                if not self.is_valid_position(next_x, next_z, goal):
                    continue
                    
                # Calculate new cost
                new_g_cost = g_costs[current_pos] + 1
                
                # If we've found a better path to this position
                if next_pos not in g_costs or new_g_cost < g_costs[next_pos]:
                    g_costs[next_pos] = new_g_cost
                    f_cost = new_g_cost + self.manhattan_distance(next_pos, goal)
                    next_node = Node(next_x, next_z, new_g_cost, f_cost)
                    heapq.heappush(open_set, next_node)
                    came_from[next_pos] = current_pos
        
        # print(f"[A*] No path found after {iterations} iterations")
        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from the came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        
        # Log path details
        # print("[A*] Path waypoints:")
        # for i, point in enumerate(path):
        #     print(f"  {i}: {point}")
        
        return path
    
    def set_known_obstacles(self, obstacles, furnace_pos=None, crafting_pos=None):
        """Set obstacles directly from mission generation"""
        self.obstacles.clear()
        
        # Store furnace and crafting table positions for later use
        self.furnace_positions = [furnace_pos] if furnace_pos else []
        self.crafting_positions = [crafting_pos] if crafting_pos else []
        
        # Add all obstacles including furnace and crafting table positions
        for x, z, block_type in obstacles:
            self.obstacles.add((x, z))
            
        # Add furnace and crafting table to obstacles
        if furnace_pos:
            self.obstacles.add(furnace_pos)
        if crafting_pos:
            self.obstacles.add(crafting_pos)
        
        print(f"[PATHFINDER] Set {len(self.obstacles)} known obstacles")
        print(f"[PATHFINDER] Known furnace at {furnace_pos}")
        print(f"[PATHFINDER] Known crafting table at {crafting_pos}")
        
    # Replace the existing path_to_crafting_table method
    def path_to_crafting_table(self, agent_host) -> Optional[List[Tuple[int, int]]]:
        """Find path to nearest crafting table using known positions"""
        try:
            obj_locs = Morgan.get_obj_locations(agent_host)
            if 'morgan' not in obj_locs:
                print("[ERROR] Could not find Morgan's position")
                return None
            
            _, curr_x, curr_z = obj_locs['morgan']
            start_pos = (curr_x, curr_z)
            
            # Use known crafting table positions if available
            if hasattr(self, 'crafting_positions') and self.crafting_positions:
                crafting_table_pos = self.crafting_positions[0]
                # print(f"[INFO] Using known crafting table position: {crafting_table_pos}")
            else:
                # Fall back to discovery method
                crafting_table_pos = self.find_nearest_block(agent_host, start_pos, "crafting_table")
                # print("[INFO] Using discovered crafting table position")
            
            if not crafting_table_pos:
                print("[ERROR] No crafting table found")
                return None
            
            # print(f"[INFO] Finding path from {start_pos} to crafting table at {crafting_table_pos}")
            return self.a_star(start_pos, crafting_table_pos)
            
        except Exception as e:
            print(f"[ERROR] Error finding path to crafting table: {e}")
            return None

    # Replace the existing path_to_furnace method
    def path_to_furnace(self, agent_host) -> Optional[List[Tuple[int, int]]]:
        """Find path to nearest furnace using known positions"""
        try:
            obj_locs = Morgan.get_obj_locations(agent_host)
            if 'morgan' not in obj_locs:
                print("[ERROR] Could not find Morgan's position")
                return None
            
            _, curr_x, curr_z = obj_locs['morgan']
            start_pos = (curr_x, curr_z)
            
            # Use known furnace positions if available
            if hasattr(self, 'furnace_positions') and self.furnace_positions:
                furnace_pos = self.furnace_positions[0]
                # print(f"[INFO] Using known furnace position: {furnace_pos}")
            else:
                # Fall back to discovery method
                furnace_pos = self.find_nearest_block(agent_host, start_pos, "furnace")
                # print("[INFO] Using discovered furnace position")
            
            if not furnace_pos:
                print("[ERROR] No furnace found")
                return None
            
            # print(f"[INFO] Finding path from {start_pos} to furnace at {furnace_pos}")
            return self.a_star(start_pos, furnace_pos)
            
        except Exception as e:
            print(f"[ERROR] Error finding path to furnace: {e}")
            return None

# Enhanced Morgan class with A* pathfinding integration
class MorganWithPathfinding:
    """
    Extension methods for Morgan class to add A* pathfinding capabilities
    """
    
    def __init__(self, morgan_agent):
        self.morgan = morgan_agent
        self.pathfinder = MorganPathfinder()
    
    def smart_move_to_crafting_table(self, agent_host):
        """Use A* to move to nearest crafting table"""
        # print("[INFO] Starting smart move to crafting table...")
        
        # Ensure the world state is updated even if we have known positions
        self.pathfinder.update_world_state(agent_host)
        
        try:
            obj_locs = Morgan.get_obj_locations(agent_host)
            if 'morgan' not in obj_locs:
                print("[ERROR] Could not find Morgan's position")
                return None
            
            _, curr_x, curr_z = obj_locs['morgan']
            start_pos = (curr_x, curr_z)
            
            # Use known crafting table positions if available
            if hasattr(self.pathfinder, 'crafting_positions') and self.pathfinder.crafting_positions:
                crafting_table_pos = self.pathfinder.crafting_positions[0]
                # print(f"[INFO] Using known crafting table position: {crafting_table_pos}")
            else:
                # Fall back to discovery method
                crafting_table_pos = self.pathfinder.find_nearest_block(agent_host, start_pos, "crafting_table")
                # print("[INFO] Using discovered crafting table position")
            
            if not crafting_table_pos:
                print("[ERROR] No crafting table found, falling back to direct movement")
                table_x, table_z = self.morgan.get_block_position(agent_host, "crafting_table")
                if table_x is not None:
                    # print(f"[INFO] Moving directly to crafting table at ({table_x}, {table_z})")
                    self.morgan.move_to(agent_host, table_x, table_z)
                return
            
            # print(f"[INFO] Finding path from {start_pos} to crafting table at {crafting_table_pos}")
            path = self.pathfinder.a_star(start_pos, crafting_table_pos)
            
            if not path:
                print("[WARN] No A* path found to crafting table, falling back to direct movement")
                table_x, table_z = self.morgan.get_block_position(agent_host, "crafting_table")
                if table_x is not None:
                    # print(f"[INFO] Moving directly to crafting table at ({table_x}, {table_z})")
                    self.morgan.move_to(agent_host, table_x, table_z)
                return
            
            # print(f"[INFO] Found A* path with {len(path)} steps: {path}")
            # Execute path - this actually moves Morgan
            self.execute_path(agent_host, path)
            
        except Exception as e:
            print(f"[ERROR] Error in smart_move_to_crafting_table: {e}")
            # Fallback to direct movement
            table_x, table_z = self.morgan.get_block_position(agent_host, "crafting_table")
            if table_x is not None:
                # print(f"[INFO] Moving directly to crafting table at ({table_x}, {table_z})")
                self.morgan.move_to(agent_host, table_x, table_z)
    
    def smart_move_to_furnace(self, agent_host):
        """Use A* to move to nearest furnace"""
        # print("[INFO] Starting smart move to furnace...")
        
        # Ensure the world state is updated even if we have known positions
        self.pathfinder.update_world_state(agent_host)
        
        try:
            obj_locs = Morgan.get_obj_locations(agent_host)
            if 'morgan' not in obj_locs:
                print("[ERROR] Could not find Morgan's position")
                return None
            
            _, curr_x, curr_z = obj_locs['morgan']
            start_pos = (curr_x, curr_z)
            
            # Use known furnace positions if available
            if hasattr(self.pathfinder, 'furnace_positions') and self.pathfinder.furnace_positions:
                furnace_pos = self.pathfinder.furnace_positions[0]
                # print(f"[INFO] Using known furnace position: {furnace_pos}")
            else:
                # Fall back to discovery method
                furnace_pos = self.pathfinder.find_nearest_block(agent_host, start_pos, "furnace")
                # print("[INFO] Using discovered furnace position")
            
            if not furnace_pos:
                print("[ERROR] No furnace found, falling back to direct movement")
                furnace_x, furnace_z = self.morgan.get_block_position(agent_host, "furnace")
                if furnace_x is not None:
                    # print(f"[INFO] Moving directly to furnace at ({furnace_x}, {furnace_z})")
                    self.morgan.move_to(agent_host, furnace_x, furnace_z)
                return
            
            # print(f"[INFO] Finding path from {start_pos} to furnace at {furnace_pos}")
            path = self.pathfinder.a_star(start_pos, furnace_pos)
            
            if not path:
                print("[WARN] No A* path found to furnace, falling back to direct movement")
                furnace_x, furnace_z = self.morgan.get_block_position(agent_host, "furnace")
                if furnace_x is not None:
                    # print(f"[INFO] Moving directly to furnace at ({furnace_x}, {furnace_z})")
                    self.morgan.move_to(agent_host, furnace_x, furnace_z)
                return
            
            # print(f"[INFO] Found A* path with {len(path)} steps: {path}")
            # Execute path - this actually moves Morgan
            self.execute_path(agent_host, path)
            
        except Exception as e:
            print(f"[ERROR] Error in smart_move_to_furnace: {e}")
            # Fallback to direct movement
            furnace_x, furnace_z = self.morgan.get_block_position(agent_host, "furnace")
            if furnace_x is not None:
                # print(f"[INFO] Moving directly to furnace at ({furnace_x}, {furnace_z})")
                self.morgan.move_to(agent_host, furnace_x, furnace_z)
    
    def execute_path(self, agent_host, path: List[Tuple[int, int]]):
        """Execute a path by moving through each waypoint with improved obstacle handling"""
        # print(f"[INFO] Executing path with {len(path)} waypoints")
        
        if not path or len(path) < 2:
            print("[WARN] Path is too short to execute")
            return
        
        # Print the full path for debugging
        path_str = " -> ".join([f"({x},{z})" for x, z in path])
        print(f"[PATH] {path_str}")
        
        # Move through each waypoint in the path
        stuck_count = 0
        last_pos = None
        
        for i, (target_x, target_z) in enumerate(path[1:], 1):  # Skip first point (current position)
            # print(f"[INFO] Moving to waypoint {i}/{len(path)-1}: ({target_x}, {target_z})")
            
            # Add a small offset to help the agent reach the center of cells
            adjusted_x = target_x + 0.5
            adjusted_z = target_z + 0.5
            
            # Get current position before movement
            obj_locs = Morgan.get_obj_locations(agent_host)
            if 'morgan' not in obj_locs:
                print("[ERROR] Could not find Morgan's position")
                continue
            
            _, curr_x, curr_z = obj_locs['morgan']
            pre_move_pos = (curr_x, curr_z)
            
            # Use Morgan's existing move_to method with a shorter timeout for intermediate points
            timeout = 8 if i == len(path)-1 else 4  # Longer timeout for final waypoint
            self.morgan.move_to(agent_host, adjusted_x, adjusted_z, timeout=timeout)
            
            # Small delay to ensure movement completes
            time.sleep(0.5)
            
            # Check if we've reached the position or got stuck
            obj_locs = Morgan.get_obj_locations(agent_host)
            if 'morgan' in obj_locs:
                _, curr_x, curr_z = obj_locs['morgan']
                post_move_pos = (curr_x, curr_z)
                
                # Calculate distance to target
                distance = math.sqrt((curr_x - target_x)**2 + (curr_z - target_z)**2)
                # print(f"[INFO] Distance to target: {distance:.2f}")
                
                # Check if we've moved at all
                movement_distance = math.sqrt(
                    (post_move_pos[0] - pre_move_pos[0])**2 + 
                    (post_move_pos[1] - pre_move_pos[1])**2
                )
                
                # Use different thresholds for intermediate vs final waypoints
                threshold = 0.8 if i == len(path)-1 else 1.5
                
                # Consider agent stuck if it hasn't moved much
                if movement_distance < 0.3:
                    stuck_count += 1
                    print(f"[WARN] Agent appears stuck (attempt {stuck_count})")
                    
                    # After multiple stuck attempts, try to find a way around the obstacle
                    if stuck_count >= 2:
                        print("[WARN] Agent stuck multiple times, marking area as obstacle")
                        
                        # Mark cells around the problematic area as obstacles
                        cell_x, cell_z = int(round(curr_x)), int(round(curr_z))
                        for dx, dz in [(0,0), (1,0), (0,1), (-1,0), (0,-1)]:
                            obstacle_x, obstacle_z = cell_x + dx, cell_z + dz
                            self.pathfinder.obstacles.add((obstacle_x, obstacle_z))
                        
                        # Recalculate path from current position to destination
                        current_pos = (int(round(curr_x)), int(round(curr_z)))
                        destination = path[-1]
                        # print(f"[INFO] Recalculating path from {current_pos} to {destination}")
                        
                        new_path = self.pathfinder.a_star(current_pos, destination)
                        if new_path and len(new_path) > 1:
                            # print(f"[INFO] Found new path with {len(new_path)} waypoints")
                            # Recursively execute the new path, reset stuck counter
                            stuck_count = 0
                            self.execute_path(agent_host, new_path)
                            return
                        else:
                            print("[ERROR] Failed to find alternative path, attempting direct movement")
                            # Try a more direct approach for the final destination
                            self.morgan.move_to(agent_host, destination[0] + 0.5, destination[1] + 0.5, timeout=10)
                            return
                    
                    # Try a small "wiggle" maneuver to unstick
                    # print("[INFO] Attempting unstick maneuver")
                    
                    # Get current orientation and try moving sideways briefly
                    self.morgan.turn_by(agent_host, 90)  # Turn right
                    self.morgan.move_distance(agent_host, 1)  # Move a bit
                    time.sleep(0.2)
                    self.morgan.move_distance(agent_host, -1)  # Move back
                    self.morgan.turn_by(agent_host, -90)  # Turn back
                    time.sleep(0.2)
                    
                    # Retry the current waypoint
                    i = max(0, i-1)  # Retry current waypoint
                    continue
                else:
                    # Reset stuck counter if we're moving
                    stuck_count = 0
                
                # If we're close enough to target, move to next waypoint
                if distance <= threshold:
                    # print(f"[INFO] Reached waypoint {i}")
                    continue
                
                # If we're not close enough, but we've moved, try again or skip
                if i < len(path)-1:  # For intermediate waypoints
                    # If we've made progress but haven't reached the target, we can continue
                    # to the next waypoint if we're getting closer to the final destination
                    if movement_distance > 0.5:
                        # print(f"[INFO] Made progress toward waypoint {i}, continuing to next")
                        continue
                else:  # For final waypoint
                    # For the final waypoint, make extra attempts to reach it exactly
                    if distance > threshold:
                        print(f"[WARN] Failed to reach final waypoint, distance: {distance:.2f}")
                        # print(f"[INFO] Making final approach to destination")
                        
                        # For known locations, use their specific approach methods
                        destination = path[-1]
                        if (hasattr(self.pathfinder, 'crafting_positions') and 
                            destination in self.pathfinder.crafting_positions):
                            # print("[INFO] Making final approach to crafting table")
                            self.morgan.approach_crafting_table(agent_host)
                        elif (hasattr(self.pathfinder, 'furnace_positions') and 
                              destination in self.pathfinder.furnace_positions):
                            # print("[INFO] Making final approach to furnace")
                            self.morgan.approach_furnace(agent_host)
                        else:
                            # One last direct movement attempt
                            self.morgan.move_to(agent_host, adjusted_x, adjusted_z, timeout=10)
        
        # Verify we actually reached the final destination
        final_x, final_z = path[-1]
        obj_locs = Morgan.get_obj_locations(agent_host)
        if 'morgan' in obj_locs:
            _, curr_x, curr_z = obj_locs['morgan']
            final_distance = math.sqrt((curr_x - final_x)**2 + (curr_z - final_z)**2)
            
            if final_distance <= 1.0:
                print(f"[SUCCESS] Path execution completed successfully. Final distance: {final_distance:.2f}")
            else:
                print(f"[WARN] Path execution completed but final distance ({final_distance:.2f}) exceeds ideal threshold.")
                
                # For known functional locations, try direct approach methods
                if (hasattr(self.pathfinder, 'crafting_positions') and 
                    (final_x, final_z) in self.pathfinder.crafting_positions):
                    # print("[INFO] Making final approach to crafting table")
                    self.morgan.approach_crafting_table(agent_host)
                elif (hasattr(self.pathfinder, 'furnace_positions') and 
                      (final_x, final_z) in self.pathfinder.furnace_positions):
                    # print("[INFO] Making final approach to furnace")
                    self.morgan.approach_furnace(agent_host)
        else:
            print("[ERROR] Could not verify final position")

# Example of how to integrate with existing Morgan class
def enhance_morgan_with_pathfinding(morgan_agent):
    """
    Factory function to create an enhanced Morgan with pathfinding
    """
    enhanced_morgan = MorganWithPathfinding(morgan_agent)
    
    # Replace the craft_item method to use smart pathfinding
    def enhanced_craft_item(agent_host, item):
        if not enhanced_morgan.morgan.can_craft(agent_host):
            # print("[INFO] Not close enough to a crafting table. Using A* pathfinding...")
            enhanced_morgan.smart_move_to_crafting_table(agent_host)
        
        # Continue with original crafting logic
        if item not in submission.food_recipes:
            print(f"[ERROR] No recipe for {item}.")
            return

        ingredients = submission.food_recipes[item]
        for item_needed in ingredients:
            if enhanced_morgan.morgan.inventory[item_needed] < ingredients.count(item_needed):
                print(f"[ERROR] Not enough {item_needed} to craft {item}.")
                return

        for item_needed in ingredients:
            enhanced_morgan.morgan.inventory[item_needed] -= 1
            enhanced_morgan.morgan.num_items_in_inv -= 1

        agent_host.sendCommand(f'craft {item}')
        enhanced_morgan.morgan.inventory[item] += 1
        enhanced_morgan.morgan.num_items_in_inv += 1
        print(f"[SUCCESS] Crafted {item}.")
        time.sleep(0.25)
    
    # Replace the cook_item method to use smart pathfinding
    def enhanced_cook_item(agent_host, cooked_item):
        if not enhanced_morgan.morgan.can_cook(agent_host):
            # print("[INFO] Not close enough to a furnace. Using A* pathfinding...")
            enhanced_morgan.smart_move_to_furnace(agent_host)

        if cooked_item not in submission.cooking_recipes:
            print(f"[ERROR] No recipe for {cooked_item}.")
            return

        ingredients = submission.cooking_recipes[cooked_item]
        for item in ingredients:
            if enhanced_morgan.morgan.inventory[item] < ingredients.count(item):
                print(f"[ERROR] Not enough {item} to cook {cooked_item}.")
                return

        for item in ingredients:
            enhanced_morgan.morgan.inventory[item] -= 1
            enhanced_morgan.morgan.num_items_in_inv -= 1

        enhanced_morgan.morgan.inventory[cooked_item] += 1
        enhanced_morgan.morgan.num_items_in_inv += 1
        print(f"[SUCCESS] Cooked {cooked_item}.")
        time.sleep(0.5)
    
    # Monkey patch the methods
    enhanced_morgan.morgan.craft_item = enhanced_craft_item
    enhanced_morgan.morgan.cook_item = enhanced_cook_item
    
    return enhanced_morgan

# Usage example:
# enhanced_morgan = enhance_morgan_with_pathfinding(your_morgan_instance)
# The enhanced Morgan will now use A* pathfinding when moving to crafting tables and furnaces