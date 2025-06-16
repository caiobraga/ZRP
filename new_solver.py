import json
import os
import time
import random
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import heapq
import logging
from math import sqrt
from itertools import combinations
from collections import defaultdict

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("zrp_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VisibilityGraphAStar:
    def __init__(self, container, obstacles):
        self.container = container
        self.obstacles = obstacles
        self.visibility_graph = {}
        self.build_visibility_graph()
        
    def _connect_point(self, point):
        """Robust point connection to visibility graph"""
        if point not in self.visibility_graph:
            self.visibility_graph[point] = {}
            
            connections = 0
            for existing_point in list(self.visibility_graph.keys()):
                if existing_point == point:
                    continue
                    
                line = LineString([point, existing_point])
                if (not any(line.crosses(obs) for obs in self.obstacles) and 
                    line.within(self.container)):
                    dist = sqrt((existing_point[0]-point[0])**2 + (existing_point[1]-point[1])**2)
                    self.visibility_graph[point][existing_point] = dist
                    self.visibility_graph[existing_point][point] = dist
                    connections += 1
            
            logger.debug(f"Connected new point {point} to {connections} existing points")
            return connections > 0  
        
        return True 

    def build_visibility_graph(self):
        """Build a visibility graph of all obstacle vertices with logging"""
        # Get all obstacle vertices
        vertices = []
        for i, obs in enumerate(self.obstacles):
            obs_vertices = list(obs.exterior.coords)
            vertices.extend(obs_vertices)
            logger.debug(f"Obstacle {i} has {len(obs_vertices)} vertices")
        
        vertices = list(set(vertices))
        logger.info(f"Total unique vertices: {len(vertices)}")
        
        logger.info("Checking visibility between vertices...")
        visible_pairs = 0
        for i, (v1, v2) in enumerate(combinations(vertices, 2)):
            line = LineString([v1, v2])
            
            crosses_obstacle = any(line.crosses(obs) or line.within(obs) for obs in self.obstacles)
            outside_container = not line.within(self.container)
            
            if not crosses_obstacle and not outside_container:
                self._add_edge(v1, v2)
                visible_pairs += 1
                
            if i % 1000 == 0 and i > 0:  
                logger.debug(f"Checked {i} vertex pairs, {visible_pairs} visible connections found")
                
        logger.info(f"Visibility graph construction complete. {visible_pairs} edges created.")
    
    def _add_edge(self, p1, p2):
        """Add an edge between two points with distance"""
        if p1 not in self.visibility_graph:
            self.visibility_graph[p1] = {}
        if p2 not in self.visibility_graph:
            self.visibility_graph[p2] = {}
            
        dist = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        self.visibility_graph[p1][p2] = dist
        self.visibility_graph[p2][p1] = dist
        logger.debug(f"Added edge: {p1} <-> {p2} (distance: {dist:.2f})")
    
    def find_path(self, start, end):
        """Robust pathfinding with proper type handling"""
        # Convert inputs to consistent format (tuples of floats)
        start = self._ensure_point_format(start)
        end = self._ensure_point_format(end)
        
        start_point = Point(start)
        end_point = Point(end)
        
        if not self.container.contains(start_point):
            logger.error(f"Start point {start} is outside container!")
            return None
            
        if not self.container.contains(end_point):
            logger.error(f"End point {end} is outside container!")
            return None
            
        for i, obs in enumerate(self.obstacles):
            if obs.contains(start_point):
                logger.error(f"Start point {start} is inside obstacle {i}!")
                return None
            if obs.contains(end_point):
                logger.error(f"End point {end} is inside obstacle {i}!")
                return None
        
        self._connect_point(start)
        self._connect_point(end)
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {p: float('inf') for p in self.visibility_graph}
        g_score[start] = 0
        f_score = {p: float('inf') for p in self.visibility_graph}
        f_score[start] = self._heuristic(start, end)
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            for neighbor, dist in self.visibility_graph[current].items():
                tentative_g = g_score[current] + dist
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None

    def _ensure_point_format(self, point):
        """Convert point to tuple of floats regardless of input format"""
        if isinstance(point, (np.ndarray, list)):
            return tuple(float(x) for x in point)
        elif isinstance(point, tuple):
            return tuple(float(x) for x in point)
        elif hasattr(point, 'x') and hasattr(point, 'y'):  
            return (float(point.x), float(point.y))
        else:
            raise ValueError(f"Cannot convert point of type {type(point)} to tuple")
    
    def _heuristic(self, p1, p2):
        """Euclidean distance heuristic"""
        return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

class OptimizedZRPTabuVisGraph:
    def __init__(self, container, obstacles, start_point, food_requirements, capacity, timeout=300):
        self.container = container
        self.obstacles = obstacles
        self.start_point = start_point
        self.food_requirements = food_requirements
        self.capacity = capacity
        self.timeout = timeout
        self.start_time = time.time()
        self.best_route = None
        self.best_length = float('inf')
        self.best_order = None
        self.best_trips = None
        
        logger.info("Initializing visibility graph...")
        self.vis_graph = VisibilityGraphAStar(container, obstacles)
        
        # Precompute obstacle representative points
        logger.info("Computing obstacle representative points...")
        self.obstacle_points = []
        for i, obs in enumerate(obstacles):
            # Use centroid as representative point
            centroid = obs.centroid
            self.obstacle_points.append((centroid.x, centroid.y))
            logger.debug(f"Obstacle {i} centroid: {self.obstacle_points[-1]}")
            
            if not container.contains(centroid):
                logger.warning(f"Obstacle {i} centroid is outside container!")
            if any(o.contains(centroid) for o in obstacles if o != obs):
                logger.warning(f"Obstacle {i} centroid is inside another obstacle!")
    
    def build_route_with_capacity(self, obstacle_order):
        """Build route with capacity constraints and multiple trips"""
        current_pos = tuple(float(x) for x in self.start_point)
        complete_route = []
        current_load = self.capacity
        touched = set()
        trips = []
        current_trip = [current_pos]
        total_distance = 0
        
        # First check if any single obstacle exceeds capacity
        for obs_idx in obstacle_order:
            if self.food_requirements[obs_idx] > self.capacity:
                logger.error(f"Obstacle {obs_idx} requires {self.food_requirements[obs_idx]}kg but capacity is only {self.capacity}kg")
                return None, set(), float('inf'), []
        
        # Split obstacle order into trips based on capacity
        trip_plans = []
        current_trip_plan = []
        current_trip_load = 0
        
        for obs_idx in obstacle_order:
            food_needed = self.food_requirements[obs_idx]
            
            if current_trip_load + food_needed > self.capacity:
                trip_plans.append(current_trip_plan)
                current_trip_plan = [obs_idx]
                current_trip_load = food_needed
            else:
                current_trip_plan.append(obs_idx)
                current_trip_load += food_needed
        
        if current_trip_plan:
            trip_plans.append(current_trip_plan)
        
        logger.debug(f"Generated trip plans: {trip_plans}")
        
        # Execute each trip plan
        for trip_idx, trip_plan in enumerate(trip_plans):
            current_trip = [current_pos]
            trip_food = sum(self.food_requirements[i] for i in trip_plan)
            logger.info(f"Starting trip {trip_idx+1} with {len(trip_plan)} obstacles ({trip_food}kg/{self.capacity}kg)")
            
            for obs_idx in trip_plan:
                obstacle = self.obstacles[obs_idx]
                target = tuple(float(x) for x in obstacle.exterior.coords[0])  # First vertex
                
                path = self.vis_graph.find_path(current_pos, target)
                if not path:
                    logger.warning(f"No path found to obstacle {obs_idx}")
                    return None, set(), float('inf'), []
                
                current_trip.extend(path[1:])
                current_pos = path[-1]
                touched.add(obs_idx)
                total_distance += self.calculate_route_length(path)
                logger.debug(f"Visited obstacle {obs_idx} (food: {self.food_requirements[obs_idx]}kg)")
            
            # Return to start after each trip
            return_path = self.vis_graph.find_path(current_pos, self.start_point)
            if not return_path:
                logger.warning("No return path found to start point")
                return None, set(), float('inf'), []
            
            current_trip.extend(return_path[1:])
            total_distance += self.calculate_route_length(return_path)
            current_pos = self.start_point
            trips.append((current_trip, set(trip_plan), trip_food))
        
        # Combine all trips into one complete route
        complete_route = []
        for trip, _, _ in trips:
            complete_route.extend(trip)
        
        return complete_route, touched, total_distance, trips
    
    def tabu_search(self, max_iterations=200, tabu_size=50, max_neighbors=10):
        logger.info("\n===== STARTING TABU SEARCH WITH CAPACITY CONSTRAINTS =====")
        num_obs = len(self.obstacles)
        current_order = list(range(num_obs))
        random.shuffle(current_order)
        tabu_list = set()
        iteration = 0

        try:
            while iteration < max_iterations:
                if time.time() - self.start_time > self.timeout:
                    raise TimeoutError()
                
                iteration += 1
                logger.info(f"\n=== Iteration {iteration} ===")
                
                # Generate neighbors by swapping two random obstacles
                neighbors = []
                for _ in range(max_neighbors):
                    i, j = random.sample(range(num_obs), 2)
                    new_order = current_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    neighbors.append(new_order)
                
                best_neighbor = None
                best_neighbor_route = None
                best_neighbor_length = float('inf')
                best_neighbor_trips = None
                valid_neighbors = 0

                for new_order in neighbors:
                    state_key = tuple(new_order)
                    
                    if state_key in tabu_list:
                        logger.debug(f"Skipping tabu neighbor: {new_order}")
                        continue
                        
                    logger.debug(f"Evaluating neighbor: {new_order}")
                    route, touched, length, trips = self.build_route_with_capacity(new_order)
                    
                    if route and len(touched) == num_obs:
                        valid_neighbors += 1
                        
                        if length < best_neighbor_length:
                            best_neighbor = new_order
                            best_neighbor_route = route
                            best_neighbor_length = length
                            best_neighbor_trips = trips
                            logger.debug(f"New best neighbor found (length: {length:.2f})")
                
                logger.info(f"Evaluated {valid_neighbors} valid neighbors this iteration")
                
                if best_neighbor:
                    current_order = best_neighbor
                    tabu_list.add(tuple(current_order))
                    if len(tabu_list) > tabu_size:
                        tabu_list.pop()
                    
                    if best_neighbor_length < self.best_length:
                        logger.info(f"NEW BEST SOLUTION FOUND: {best_neighbor_length:.2f}")
                        self.best_length = best_neighbor_length
                        self.best_route = best_neighbor_route
                        self.best_order = current_order
                        self.best_trips = best_neighbor_trips
                else:
                    logger.info("No improving neighbors found this iteration")
                
                if iteration % 20 == 0 and iteration > 0:
                    logger.info("Diversifying search with random shuffle")
                    current_order = list(range(num_obs))
                    random.shuffle(current_order)

        except TimeoutError:
            logger.warning("Timeout reached! Returning best found solution...")

        logger.info("\n===== TABU SEARCH COMPLETE =====")
        logger.info(f"Best solution length: {self.best_length:.2f}")
        logger.info(f"Best solution order: {self.best_order}")
        logger.info(f"Number of trips required: {len(self.best_trips) if self.best_trips else 0}")
        
        if self.best_trips:
            for i, (_, obs_covered, trip_food) in enumerate(self.best_trips):
                logger.info(f"Trip {i+1}:")
                logger.info(f"  - Obstacles: {sorted(obs_covered)}")
                logger.info(f"  - Total food: {trip_food}kg/{self.capacity}kg")
                logger.info(f"  - Food per obstacle: {', '.join(f'{obs}:{self.food_requirements[obs]}kg' for obs in sorted(obs_covered))}")
        
        return self.best_route
    
    def calculate_route_length(self, route):
        if len(route) < 2:
            return float('inf')
        return LineString(route).length
    
    def plot_solution(self):
        if not self.best_route or not self.best_trips:
            logger.error("No valid route to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot container
        x, y = self.container.exterior.xy
        plt.plot(x, y, 'b-', linewidth=3, label='Container')
        
        # Plot obstacles with food requirements
        for i, poly in enumerate(self.obstacles):
            x, y = poly.exterior.xy
            plt.fill(x, y, 'red', alpha=0.3)
            # Plot obstacle numbers and food requirements
            centroid = poly.centroid
            plt.text(centroid.x, centroid.y, f"{i}\n({self.food_requirements[i]}kg)", 
                    ha='center', va='center', color='black', fontsize=8)
        
        # Plot each trip with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.best_trips)))
        for idx, (trip_route, obs_covered, trip_food) in enumerate(self.best_trips):
            rx, ry = zip(*trip_route)
            plt.plot(rx, ry, '-', color=colors[idx], linewidth=2, 
                    label=f'Trip {idx+1} ({trip_food}kg/{self.capacity}kg)')
            plt.scatter(rx, ry, color=colors[idx], s=20)
        
        # Plot start point
        plt.scatter(self.start_point[0], self.start_point[1], 
                  color='blue', s=100, marker='o', label='Start/Refill')
        
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.title(f"Best Solution (Length: {self.best_length:.2f}, Trips: {len(self.best_trips)})")
        plt.show()

def get_instance_path(filename):
    """Get absolute path to instance file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(script_dir, "instances")
    
    # Check if instances directory exists
    if not os.path.exists(instances_dir):
        os.makedirs(instances_dir)
        logger.warning(f"Created instances directory at {instances_dir}")
    
    instance_path = os.path.join(instances_dir, filename)
    
    if not os.path.exists(instance_path):
        raise FileNotFoundError(
            f"Instance file not found at {instance_path}\n"
            f"Please ensure the file exists in the instances directory."
        )
    
    return instance_path


if __name__ == "__main__":
    try:
        # Get the correct path to the instance file
        instance_path = get_instance_path("berlin_zoo_instance.json")
        logger.info(f"Loading instance from: {instance_path}")
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        container = Polygon(data['container']['coordinates'][0])
        obstacles = [Polygon(obs['coordinates'][0]) for obs in data['obstacles']]
        start_point = tuple(data['start_point'])
        
        # Get food requirements and capacity
        food_requirements = data['food_requirements']
        capacity = data['capacity']
        
        logger.info(f"Loaded instance with {len(obstacles)} obstacles")
        logger.info(f"Food requirements: {food_requirements}")
        logger.info(f"Zookeeper capacity: {capacity}kg")
        
        # Check if any obstacle exceeds capacity
        for i, req in enumerate(food_requirements):
            if req > capacity:
                logger.warning(f"Obstacle {i} requires {req}kg which exceeds capacity of {capacity}kg")
        
        # Initialize solver with capacity constraints
        solver = OptimizedZRPTabuVisGraph(
            container, 
            obstacles, 
            start_point, 
            food_requirements, 
            capacity,
            timeout=300
        )
        
        # Run tabu search with progress reporting
        logger.info("Starting path optimization with capacity constraints...")
        best_route = solver.tabu_search(max_iterations=200, tabu_size=50, max_neighbors=10)
        
        if best_route:
            logger.info(f"Optimization complete. Best route length: {LineString(best_route).length:.2f}")
            solver.plot_solution()
        else:
            logger.error("No valid route found after optimization")
            
    except FileNotFoundError as e:
        logger.error(str(e))
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in instance file")
    except KeyError as e:
        logger.error(f"Missing required field in JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)