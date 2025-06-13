import random
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from itertools import product, combinations, permutations
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'zrp_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
class ZRPAlgorithm:
    def __init__(self, container_polygon, obstacle_polygons, start_point=None, epsilon2=0.1, epsilon3=0.1):
        self.Π = container_polygon
        self.P = obstacle_polygons
        self.ε2 = epsilon2
        self.ε3 = epsilon3
        self.route = None
        self.length_history = []
        self.best_route = None
        self.best_length = float('inf')
        
        # Set start point (random if not provided)
        if start_point is None:
            self.start_point = self.generate_random_start_point()
        else:
            self.start_point = start_point
            
        self.all_candidate_points = self.generate_all_candidate_points()
        self.best_zrp_pairs = None
        
    def generate_random_start_point(self):
        """Generate a random start point inside container but outside obstacles"""
        while True:
            # Get container bounds
            min_x, min_y, max_x, max_y = self.Π.bounds
            
            # Generate random point
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            point = Point(x, y)
            
            # Check if point is inside container and outside all obstacles
            if (self.Π.contains(point) and 
                all(not poly.contains(point) for poly in self.P)):
                return (x, y)
    
    def generate_all_candidate_points(self):
        """Generate all potential entry points (vertices + edge midpoints) for each obstacle"""
        all_points = []
        for poly in self.P:
            # Get all vertices and edge midpoints
            vertices = list(poly.exterior.coords)[:-1]  # Exclude repeated closing point
            edges = list(zip(vertices, vertices[1:] + [vertices[0]]))
            edge_midpoints = [((e[0][0]+e[1][0])/2, (e[0][1]+e[1][1])/2) for e in edges]
            
            # All candidate points (vertices + midpoints)
            all_points.append(vertices + edge_midpoints)
        return all_points
    
    def evaluate_route(self, zrp_pairs):
        """Evaluate a complete route using given entry points (same entry and exit)"""
        route = [self.start_point]
        
        # Find optimal obstacle order for these points
        obstacle_order = self.find_optimal_obstacle_order(zrp_pairs)
        
        # Connect to first obstacle's entry point
        first_entry = zrp_pairs[obstacle_order[0]][0]
        path_to_first = self.extended_shortest_path(self.start_point, first_entry, self.P)
        route.extend(path_to_first[1:])
        
        # Connect through all obstacles in chosen order
        for i in range(len(obstacle_order)):
            current_idx = obstacle_order[i]
            entry_point = zrp_pairs[current_idx][0]
            route.append(entry_point)
            
            if i < len(obstacle_order)-1:
                next_idx = obstacle_order[i+1]
                next_entry = zrp_pairs[next_idx][0]
                path_segment = self.extended_shortest_path(entry_point, next_entry, 
                                                         [p for j, p in enumerate(self.P) 
                                                          if j not in [current_idx, next_idx]])
                route.extend(path_segment[1:])
        
        # Return to start point
        if len(obstacle_order) > 0:
            last_entry = zrp_pairs[obstacle_order[-1]][0]
            closing_segment = self.extended_shortest_path(last_entry, self.start_point,
                                                         [p for j, p in enumerate(self.P) 
                                                          if j != obstacle_order[-1]])
            route.extend(closing_segment[1:])
        
        return route, LineString(route).length
    
    def find_optimal_obstacle_order(self, zrp_pairs):
        """Improved obstacle ordering that considers overall route length"""
        if not zrp_pairs:
            return []
        
        # Create distance matrix between all pairs
        n = len(zrp_pairs)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i,j] = Point(zrp_pairs[i][0]).distance(Point(zrp_pairs[j][0]))
        
        # Use nearest neighbor with lookahead
        unvisited = set(range(n))
        order = []
        
        # Start with point closest to start
        start_dists = [Point(self.start_point).distance(Point(pair[0])) for pair in zrp_pairs]
        current = np.argmin(start_dists)
        order.append(current)
        unvisited.remove(current)
        
        while unvisited:
            # Find next point considering both current distance and connection to remaining points
            best_next = None
            best_score = float('inf')
            
            for candidate in unvisited:
                # Score based on distance to candidate plus minimal distance from candidate to remaining
                dist_to_candidate = dist_matrix[current, candidate]
                
                if len(unvisited) > 1:
                    min_from_candidate = min(dist_matrix[candidate, other] 
                                        for other in unvisited if other != candidate)
                    score = dist_to_candidate + 0.5 * min_from_candidate
                else:
                    score = dist_to_candidate
                
                if score < best_score:
                    best_score = score
                    best_next = candidate
            
            order.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        return order
    
  
        
    def calculate_zrp_pairs(self):
        """Make all vertices and edge midpoints potential entry/exit points"""
        zrp_pairs = []
        for i, poly in enumerate(self.P):
            # Get all vertices and edge midpoints
            vertices = list(poly.exterior.coords)[:-1]  # Exclude repeated closing point
            edges = list(zip(vertices, vertices[1:] + [vertices[0]]))
            edge_midpoints = [((e[0][0]+e[1][0])/2, (e[0][1]+e[1][1])/2) for e in edges]
            
            # All candidate points (vertices + midpoints)
            candidate_points = vertices + edge_midpoints
            
            # For each polygon, select two distinct points as entry/exit
            # We'll choose points that face adjacent polygons
            prev_poly = self.P[i-1] if i > 0 else self.P[-1]
            next_poly = self.P[i+1] if i < len(self.P)-1 else self.P[0]
            
            # Find best entry (facing previous polygon)
            entry = self.select_best_facing_point(candidate_points, poly, prev_poly)
            
            # Find best exit (facing next polygon)
            exit_pt = self.select_best_facing_point(candidate_points, poly, next_poly)
            
            # Ensure they're different points
            if entry == exit_pt:
                # If same point, choose another candidate
                other_candidates = [p for p in candidate_points if p != entry]
                if other_candidates:
                    exit_pt = random.choice(other_candidates)
            
            zrp_pairs.append((entry, exit_pt))
        return zrp_pairs
    
    def select_best_facing_point(self, candidates, poly, target_poly):
        """Select candidate point that best faces target polygon"""
        poly_center = np.array(poly.centroid.coords[0])
        target_center = np.array(target_poly.centroid.coords[0])
        direction = target_center - poly_center
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        max_score = -np.inf
        best_point = None
        
        for point in candidates:
            point_vec = np.array(point) - poly_center
            point_vec = point_vec / (np.linalg.norm(point_vec) + 1e-8)
            
            # Score based on direction alignment and distance
            score = np.dot(point_vec, direction) + 0.3/(1 + Point(point).distance(Point(target_center)))
            if score > max_score:
                max_score = score
                best_point = point
                
        return best_point

    def generate_star_polygon(self, center, radius, points, irregularity=0.5):
        """Generate interesting star-shaped polygons"""
        angles = np.linspace(0, 2*np.pi, points, endpoint=False)
        
        # Add irregularity
        angles += np.random.uniform(-irregularity, irregularity, points)
        
        # Generate radii
        radii = radius * (1 + np.random.uniform(-0.3, 0.3, points))
        
        # Calculate vertices
        vertices = []
        for angle, r in zip(angles, radii):
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            vertices.append((x, y))
        
        # Ensure convex hull to avoid self-intersections
        hull = ConvexHull(vertices)
        ordered_vertices = [vertices[i] for i in hull.vertices]
        
        return Polygon(ordered_vertices)

    
    
    def project_to_boundary(self, point, polygon):
        """Project point to polygon boundary"""
        boundary = polygon.exterior
        nearest = boundary.interpolate(boundary.project(Point(point)))
        return nearest.coords[0]
    
    def find_facing_point(self, poly, target_poly):
        """Find optimal boundary point facing target polygon"""
        poly_center = np.array(poly.centroid.coords[0])
        target_center = np.array(target_poly.centroid.coords[0])
        direction = target_center - poly_center
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        max_score = -np.inf
        best_point = None
        
        for dist in np.linspace(0, poly.exterior.length, 100):
            point = poly.exterior.interpolate(dist)
            point_vec = np.array(point.coords[0]) - poly_center
            point_vec = point_vec / (np.linalg.norm(point_vec) + 1e-8)
            
            # Score based on direction alignment and distance
            score = np.dot(point_vec, direction) + 0.3/(1 + Point(point).distance(Point(target_center)))
            if score > max_score:
                max_score = score
                best_point = point.coords[0]
                
        return best_point

    def shrink_polygon(self, polygon, epsilon):
        """Create smaller polygon with validation"""
        shrunk = polygon.buffer(-epsilon, resolution=32, join_style=2)
        return shrunk if shrunk.is_valid else polygon

    def extended_shortest_path(self, p1, p2, obstacles):
        """Improved path finding that better navigates around obstacles"""
        # First try direct path
        direct_line = LineString([p1, p2])
        valid = True
        for obs in obstacles:
            if direct_line.crosses(obs):
                valid = False
                break
        if valid:
            return [p1, p2]
        
        # If direct path crosses obstacles, find path around them
        path = [p1]
        remaining_obstacles = [obs for obs in obstacles if direct_line.crosses(obs)]
        
        # Sort obstacles by distance from p1
        remaining_obstacles.sort(key=lambda obs: Point(p1).distance(obs))
        
        for obs in remaining_obstacles:
            # Find closest points on obstacle to current position and target
            boundary = obs.exterior
            closest_to_current = boundary.interpolate(boundary.project(Point(path[-1])))
            closest_to_target = boundary.interpolate(boundary.project(Point(p2)))
            
            # Get all boundary points
            hull_points = list(obs.exterior.coords)
            
            # Find indices of closest points
            idx1 = min(range(len(hull_points)),
                    key=lambda i: Point(hull_points[i]).distance(closest_to_current))
            idx2 = min(range(len(hull_points)),
                    key=lambda i: Point(hull_points[i]).distance(closest_to_target))
            
            # Add path around obstacle
            if idx1 < idx2:
                path.extend(hull_points[idx1:idx2+1])
            else:
                path.extend(hull_points[idx2:idx1+1][::-1])
        
        path.append(p2)
        return path
    
    def find_tangent(self, point, polygon):
        """Find closest point on polygon to external point"""
        point = Point(point)
        boundary = polygon.exterior
        closest = boundary.interpolate(boundary.project(point))
        return closest.coords[0]

    def procedure4(self, P1, P2, P3, p11, p12, p21, p22, p31, p32):
        """Optimize route segment through three consecutive polygons"""
        P2_shrunk = self.shrink_polygon(P2, self.ε3)
        if not P2_shrunk.is_valid:
            return None

        # Calculate paths avoiding other obstacles
        V12 = self.extended_shortest_path(p12, p21, [p for p in self.P if p not in [P1, P2]])
        V23 = self.extended_shortest_path(p22, p31, [p for p in self.P if p not in [P2, P3]])
        
        if len(V12) < 2 or len(V23) < 2:
            return None

        # Find entry and exit points on shrunk polygon
        q1 = self.find_intersection(V12, P2_shrunk, first=True)
        q3 = self.find_intersection(V23, P2_shrunk, first=False)
        
        if q1 is None or q3 is None:
            return None

        # Check if direct connection crosses shrunk polygon
        if LineString([q1, q3]).crosses(P2_shrunk):
            # Find path around P2_shrunk
            hull_points = list(P2_shrunk.exterior.coords)
            
            t1 = self.find_tangent(q1, P2_shrunk)
            t2 = self.find_tangent(q3, P2_shrunk)
            
            idx1 = min(range(len(hull_points)),
                      key=lambda i: np.linalg.norm(np.array(hull_points[i]) - np.array(t1)))
            idx2 = min(range(len(hull_points)),
                      key=lambda i: np.linalg.norm(np.array(hull_points[i]) - np.array(t2)))
            
            if idx1 < idx2:
                path = hull_points[idx1:idx2+1]
            else:
                path = hull_points[idx2:idx1+1][::-1]
            
            refined_path = [q1] + path + [q3]
        else:
            refined_path = [q1, q3]
        
        return V12[:-1] + refined_path + V23[1:]

    def find_intersection(self, path, polygon, first=True):
        """Find intersection points between path and polygon"""
        for i in range(len(path)-1):
            idx = i if first else len(path)-2-i
            segment = LineString([path[idx], path[idx+1]])
            intersection = segment.intersection(polygon)
            
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    return (intersection.x, intersection.y)
                elif intersection.geom_type == 'MultiPoint':
                    return (intersection[0].x, intersection[0].y) if first else (intersection[-1].x, intersection[-1].y)
        return None
    
    def validate_route(self, route=None):
        """More detailed route validation"""
        route_to_check = route if route is not None else self.route
        if not route_to_check or len(route_to_check) < 2:
            return False
            
        route_line = LineString(route_to_check)
        
        # Check container boundary
        if not self.Π.contains(route_line):
            print("Route leaves zoo boundary")
            return False
        
        # Check obstacles
        valid = True
        for i, poly in enumerate(self.P):
            if route_line.crosses(poly):
                print(f"Route crosses enclosure {i} at:")
                # Find where it crosses
                intersection = route_line.intersection(poly)
                if intersection.geom_type == 'Point':
                    print(f"  - Point ({intersection.x:.2f}, {intersection.y:.2f})")
                elif intersection.geom_type == 'MultiPoint':
                    for pt in intersection:
                        print(f"  - Point ({pt.x:.2f}, {pt.y:.2f})")
                valid = False
        
        return valid
    
    def repair_route(self):
        """More robust route repair that ensures valid path"""
        if not self.route:
            return False
            
        new_route = [self.route[0]]
        
        for i in range(1, len(self.route)):
            current_point = new_route[-1]
            next_point = self.route[i]
            
            # Find path avoiding ALL obstacles
            repaired_segment = self.extended_shortest_path(current_point, next_point, self.P)
            
            if len(repaired_segment) > 1:
                new_route.extend(repaired_segment[1:])
            else:
                new_route.append(next_point)
        
        self.route = new_route
        return self.validate_route(strict=True)
    
    def algorithm5(self, max_combinations=100, max_iterations=30, improvement_threshold=0.5):
            """Enhanced main optimization algorithm with detailed logging"""
            logging.info("===== Starting Route Optimization =====")
            logging.info(f"Parameters: max_combinations={max_combinations}, max_iterations={max_iterations}")
            
            # Phase 1: Try with calculated zrp pairs
            logging.info("Phase 1: Trying calculated ZRP pairs...")
            calculated_pairs = self.calculate_zrp_pairs()
            logging.info(f"Calculated {len(calculated_pairs)} initial touching point pairs")
            
            route, length = self.evaluate_route(calculated_pairs)
            if route:
                logging.info(f"Initial route length: {length:.2f}")
                if self.validate_route(route):
                    self.best_zrp_pairs = calculated_pairs
                    self.route = route
                    self.best_route = route
                    self.best_length = length
                    logging.info("Success! Initial route is valid")
                    #return route
                else:
                    logging.warning("Initial route failed validation")
            else:
                logging.warning("Failed to generate initial route")

            # Phase 2: Optimized search
            logging.info("Phase 2: Starting optimized search...")
            best_pairs, best_route, best_length = self.find_optimal_zrp_pairs(max_combinations)
            
            if best_pairs is None:
                logging.warning("No valid pairs found in optimized search - attempting repair")
                self.route = route if route else []
                if self.repair_route():
                    logging.info("Repair successful")
                    return self.route
                else:
                    logging.error("Repair failed")
                    return None
            
            # Phase 3: Refinement
            logging.info("Phase 3: Route refinement...")
            self.best_zrp_pairs = best_pairs
            self.route = best_route
            self.best_route = best_route
            self.best_length = best_length
            
            logging.info(f"Best route found with length: {best_length:.2f}")
            
            # Additional optimization passes
            improvement = float('inf')
            iteration = 0

            while iteration < max_iterations and improvement > improvement_threshold:
                iteration += 1
                logging.info(f"\nOptimization pass {iteration}/{max_iterations}")
                
                # Try local improvements
                new_pairs, new_length = self.optimize_zrp_pairs_sa(
                    initial_temp=1000/(iteration+1),
                    cooling_rate=0.95,
                    iterations=50
                )
                
                if new_length < best_length:
                    improvement = best_length - new_length
                    best_length = new_length
                    best_pairs = new_pairs
                    best_route, _ = self.evaluate_route(best_pairs)
                    
                    logging.info(f"Improved route length to {best_length:.2f} (Δ={improvement:.2f})")
                    
                    # Use non-strict validation during optimization
                    if self.validate_route(best_route, strict=False):
                        self.best_zrp_pairs = best_pairs
                        self.route = best_route
                        self.best_route = best_route
                        self.best_length = best_length
                    else:
                        logging.warning("Improved route failed validation - attempting repair")
                        self.route = best_route
                        if self.repair_route():
                            best_length = LineString(self.route).length
                            logging.info(f"Repaired route length: {best_length:.2f}")
                else:
                    improvement = 0
                    logging.info("No improvement found in this pass")
            
            if self.best_route:
                final_length = LineString(self.best_route).length
                logging.info(f"Final optimized route length: {final_length:.2f}")
                logging.info("Route validation checks:")
                
                # Detailed validation logging
                if not self.validate_route(self.best_route):
                    logging.error("Final route validation failed!")
                else:
                    logging.info("All validation checks passed")
                    
                return self.best_route
            
            logging.error("Failed to find valid route after all optimization attempts")
            return None

    def optimize_zrp_pairs_sa(self, initial_temp=1000, cooling_rate=0.95, iterations=100):
        """Optimize ZRP pairs using simulated annealing"""
        current_pairs = self.best_zrp_pairs.copy()
        current_length = self.best_length
        temp = initial_temp
        
        for i in range(iterations):
            # Create neighbor solution by perturbing one point
            new_pairs = current_pairs.copy()
            idx = random.randint(0, len(self.P)-1)
            candidates = self.all_candidate_points[idx]
            new_point = random.choice(candidates)
            new_pairs[idx] = (new_point, new_point)  # Using same entry/exit for simplicity
            
            # Evaluate new solution
            new_route, new_length = self.evaluate_route(new_pairs)
            
            # Acceptance criteria
            if new_route and (new_length < current_length or 
                            random.random() < np.exp((current_length - new_length)/temp)):
                current_pairs = new_pairs
                current_length = new_length
                
            # Cool down
            temp *= cooling_rate
            
        return current_pairs, current_length

    def find_optimal_zrp_pairs(self, max_combinations=100):
        """Find optimal entry points with detailed logging"""
        logging.info(f"Searching for optimal ZRP pairs (max {max_combinations} combinations)")
        
        best_pairs = None
        best_length = float('inf')
        best_route = None
        num_obstacles = len(self.P)
        
        logging.info(f"Evaluating up to {max_combinations} combinations across {num_obstacles} obstacles")
        
        # Generate candidate points
        candidate_indices = []
        for i in range(num_obstacles):
            num_candidates = len(self.all_candidate_points[i])
            max_per_obstacle = min(5, max(1, int(max_combinations ** (1/num_obstacles))))
            candidate_indices.append(list(range(min(max_per_obstacle, num_candidates))))
            logging.debug(f"Obstacle {i}: evaluating {len(candidate_indices[-1])}/{num_candidates} points")
        
        # Evaluate combinations
        evaluated = 0
        best_progress = 0
        
        for point_indices in product(*candidate_indices):
            if evaluated >= max_combinations:
                break
                
            evaluated += 1
            progress = int(100 * evaluated / max_combinations)
            if progress > best_progress:
                best_progress = progress
                logging.info(f"Progress: {progress}% ({evaluated}/{max_combinations})")
            
            # Create pairs
            for _ in range(max_combinations):
                # Generate random ZRP pairs
                zrp_pairs = []
                for i in range(len(self.P)):
                    candidates = self.all_candidate_points[i]
                    point = random.choice(candidates)
                    zrp_pairs.append((point, point))
                
                route, length = self.evaluate_route(zrp_pairs)
                
                if route and length < best_length:
                    self.route = route
                    if self.repair_route():  # Try to repair first
                        repaired_length = LineString(self.route).length
                        if repaired_length < best_length:
                            best_length = repaired_length
                            best_pairs = zrp_pairs
                            best_route = self.route
                        else:
                            logging.debug(f"Found shorter route ({length:.2f}) but failed validation")
        
        if best_pairs:
            logging.info(f"Best solution found after {evaluated} evaluations")
            logging.info(f"Best length: {best_length:.2f}")
        else:
            logging.warning("No valid pairs found in search")
            
        return best_pairs, best_route, best_length

    def validate_route(self, route=None, strict=False):
        """Enhanced validation with strict mode option"""
        route_to_check = route if route is not None else self.route
        if not route_to_check or len(route_to_check) < 2:
            return False
            
        route_line = LineString(route_to_check)
        
        # Always check container boundary
        if not self.Π.contains(route_line):
            return False
        
        # In non-strict mode, allow some crossings that might be fixed later
        if strict:
            for poly in self.P:
                if route_line.crosses(poly):
                    return False
        
        # Check all obstacles are touched
        touched = [False] * len(self.P)
        for point in route_to_check:
            for i, poly in enumerate(self.P):
                if Point(point).touches(poly):
                    touched[i] = True
                    break
        
        return all(touched)
    
    def plot_environment(self):
        """Just plot the environment without any solution"""
        plt.figure(figsize=(12, 8))
        
        # Plot container
        x, y = self.Π.exterior.xy
        plt.plot(x, y, 'b-', linewidth=3, label='Zoo Boundary')
        plt.fill(x, y, 'b', alpha=0.1)
        
        # Plot obstacles
        for i, poly in enumerate(self.P):
            x, y = poly.exterior.xy
            plt.fill(x, y, 'r', alpha=0.3)
            plt.plot(x, y, 'r-', linewidth=2)
            plt.text(poly.centroid.x, poly.centroid.y, f'Enclosure {i}', 
                    ha='center', va='center', color='darkred')
        
        # Plot start point
        plt.plot(self.start_point[0], self.start_point[1], 'k*', 
                markersize=15, markeredgecolor='w', label='Zookeeper Station')
        
        plt.legend()
        plt.title('Zoo Environment')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
    def plot_solution(self):
        """Visualize the Zookeeper Route Problem solution"""
        plt.figure(figsize=(12, 8))
        
        # Plot container (zoo)
        x, y = self.Π.exterior.xy
        plt.plot(x, y, 'b-', linewidth=3, label='Zoo Boundary')
        plt.fill(x, y, 'b', alpha=0.1)
        
        # Plot obstacles (animal enclosures)
        for i, poly in enumerate(self.P):
            x, y = poly.exterior.xy
            plt.fill(x, y, 'r', alpha=0.3)
            plt.plot(x, y, 'r-', linewidth=2)
            plt.text(poly.centroid.x, poly.centroid.y, f'Enclosure {i}', 
                    ha='center', va='center', color='darkred')
        
        # Plot candidate points (potential touching points)
        for i, points in enumerate(self.all_candidate_points):
            px, py = zip(*points)
            plt.plot(px, py, 'yo', markersize=6, alpha=0.5, 
                    label='Candidate Points' if i == 0 else "")
        
        # Plot selected entry points (where zookeeper touches enclosures)
        if hasattr(self, 'best_zrp_pairs') and self.best_zrp_pairs:
            for i, (entry, exit_) in enumerate(self.best_zrp_pairs):
                plt.plot(entry[0], entry[1], 'go', markersize=10, 
                        label='Entry Points' if i == 0 else "")
                plt.plot(exit_[0], exit_[1], 'co', markersize=10,
                        label='Exit Points' if i == 0 else "")
                plt.text(entry[0], entry[1], f'E{i}', ha='right', va='bottom')
                plt.text(exit_[0], exit_[1], f'X{i}', ha='left', va='bottom')
        
        # Plot start/end point (zookeeper's station)
        plt.plot(self.start_point[0], self.start_point[1], 'k*', 
                markersize=15, markeredgecolor='w', label='Zookeeper Station')
        
        # Plot the route if available
        if hasattr(self, 'route') and self.route and len(self.route) > 1:
            route_line = LineString(self.route)
            if self.validate_route():
                # Draw valid route in green
                route_x, route_y = zip(*self.route)
                plt.plot(route_x, route_y, 'g-', linewidth=2, label='Zookeeper Route')
                
                # Add direction arrows
                for i in range(0, len(self.route)-1, max(1, len(self.route)//10)):
                    dx = self.route[i+1][0] - self.route[i][0]
                    dy = self.route[i+1][1] - self.route[i][1]
                    plt.arrow(self.route[i][0], self.route[i][1], 
                            dx*0.8, dy*0.8, head_width=0.2, 
                            head_length=0.3, fc='green', ec='green')
                
                plt.title(f'Valid Zookeeper Route - Length: {route_line.length:.2f}')
            else:
                # Draw invalid route in red
                route_x, route_y = zip(*self.route)
                plt.plot(route_x, route_y, 'r-', linewidth=2, label='INVALID Route')
                plt.title('INVALID ROUTE - Crosses Enclosures', color='red')
        else:
            plt.title('No valid route found', color='red')
        
        plt.legend(loc='upper right')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def build_initial_route(self):
        """Build initial route connecting all ZRP pairs"""
        if not hasattr(self, 'best_zrp_pairs') or not self.best_zrp_pairs:
            return
            
        self.route = []
        for i, (p1, p2) in enumerate(self.best_zrp_pairs):
            self.route.append(p1)
            if i < len(self.best_zrp_pairs)-1:
                next_p1 = self.best_zrp_pairs[i+1][0]
                path_segment = self.extended_shortest_path(p2, next_p1, 
                                                        [p for p in self.P if p not in [self.P[i], self.P[i+1]]])
                self.route.extend(path_segment[1:])
        
        # Close the loop
        if len(self.best_zrp_pairs) > 1:
            last_p2 = self.best_zrp_pairs[-1][1]
            first_p1 = self.best_zrp_pairs[0][0]
            closing_segment = self.extended_shortest_path(last_p2, first_p1,
                                                        [p for p in self.P if p not in [self.P[-1], self.P[0]]])
            self.route.extend(closing_segment[1:])
    
    def validate_segment(self, segment):
        """Validate a route segment doesn't cross obstacles"""
        segment_line = LineString(segment)
        for poly in self.P:
            if segment_line.crosses(poly):
                return False
        return True
    
    def find_segment_indices(self, route, segment):
        """Find start and end indices of segment in route"""
        if len(segment) < 2:
            return None, None
            
        # Find first point
        start_idx = next((i for i, p in enumerate(route) 
                         if np.allclose(p, segment[0], atol=1e-2)), None)
        if start_idx is None:
            return None, None
            
        # Find last point after start_idx
        end_idx = next((i for i, p in enumerate(route[start_idx:], start_idx)
                       if np.allclose(p, segment[-1], atol=1e-2)), None)
        
        return start_idx, end_idx
    
    
    def plot_solution(self):
        """Visualize the Zookeeper Route Problem solution"""
        plt.figure(figsize=(12, 8))
        
        # Plot container (zoo boundary)
        x, y = self.Π.exterior.xy
        plt.plot(x, y, 'b-', linewidth=3, label='Zoo Boundary')
        plt.fill(x, y, 'b', alpha=0.1)
        
        # Plot obstacles (animal enclosures)
        for i, poly in enumerate(self.P):
            x, y = poly.exterior.xy
            plt.fill(x, y, 'r', alpha=0.3)
            plt.plot(x, y, 'r-', linewidth=2)
            plt.text(poly.centroid.x, poly.centroid.y, f'Enclosure {i}', 
                    ha='center', va='center', color='darkred')
        
        # Plot candidate points (potential touching points)
        for i, points in enumerate(self.all_candidate_points):
            px, py = zip(*points)
            plt.plot(px, py, 'yo', markersize=6, alpha=0.5, 
                    label='Candidate Points' if i == 0 else "")
        
        # Plot selected entry/exit points if they exist
        if hasattr(self, 'best_zrp_pairs') and self.best_zrp_pairs:
            for i, (entry, exit_) in enumerate(self.best_zrp_pairs):
                plt.plot(entry[0], entry[1], 'go', markersize=10, 
                        label='Entry Points' if i == 0 else "")
                plt.plot(exit_[0], exit_[1], 'co', markersize=10,
                        label='Exit Points' if i == 0 else "")
                plt.text(entry[0], entry[1], f'E{i}', ha='right', va='bottom')
                plt.text(exit_[0], exit_[1], f'X{i}', ha='left', va='bottom')
        
        # Plot start/end point (zookeeper's station)
        plt.plot(self.start_point[0], self.start_point[1], 'k*', 
                markersize=15, markeredgecolor='w', label='Zookeeper Station')
        
        # Plot the route if available
        if hasattr(self, 'route') and self.route and len(self.route) > 1:
            if self.validate_route():
                # Valid route (green)
                route_x, route_y = zip(*self.route)
                plt.plot(route_x, route_y, 'g-', linewidth=2, label='Zookeeper Route')
                
                # Add direction arrows
                for i in range(0, len(self.route)-1, max(1, len(self.route)//10)):
                    dx = self.route[i+1][0] - self.route[i][0]
                    dy = self.route[i+1][1] - self.route[i][1]
                    plt.arrow(self.route[i][0], self.route[i][1], 
                            dx*0.8, dy*0.8, head_width=0.2, 
                            head_length=0.3, fc='green', ec='green')
                
                plt.title(f'Valid Zookeeper Route - Length: {LineString(self.route).length:.2f}')
            else:
                # Invalid route (red)
                route_x, route_y = zip(*self.route)
                plt.plot(route_x, route_y, 'r-', linewidth=2, label='INVALID Route')
                plt.title('INVALID ROUTE - Crosses Enclosures', color='red')
        else:
            plt.title('No valid route found', color='red')
        
        plt.legend(loc='upper right')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def create_interesting_environment():
        """Create container with interesting obstacle shapes"""
        container = Polygon([(0, 0), (20, 0), (20, 15), (0, 15)])
        
        # Generate interesting obstacle shapes
        obstacles = []
        
        # Star-shaped polygons
        obstacles.append(Polygon([(3, 3), (5, 2), (7, 3), (6, 5), (7, 7), (5, 6), (3, 7), (4, 5)]))
        obstacles.append(Polygon([(12, 4), (14, 2), (16, 4), (17, 6), (15, 7), (13, 6), (12, 5)]))
        
        # L-shaped polygon
        obstacles.append(Polygon([(5, 10), (8, 10), (8, 12), (10, 12), (10, 8), (5, 8)]))
        
        # Plus-shaped polygon
        obstacles.append(Polygon([(14, 9), (16, 9), (16, 11), (18, 11), (18, 13), 
                                (16, 13), (16, 15), (14, 15), (14, 13), (12, 13), 
                                (12, 11), (14, 11)]))
        
        # Random convex polygon
        random_points = [(2 + random.random()*3, 10 + random.random()*3) for _ in range(6)]
        hull = ConvexHull(random_points)
        obstacles.append(Polygon([random_points[i] for i in hull.vertices]))
        
        return container, obstacles

if __name__ == "__main__":
    container, obstacles = create_interesting_environment()
    zrp = ZRPAlgorithm(container, obstacles, epsilon2=0.1, epsilon3=0.1)
    
    # First visualize the environment
    zrp.plot_environment()
    
    # Try to solve
    final_route = zrp.algorithm5(max_combinations=200)  # Increased from 100
    
    if final_route:
        print(f"Success! Route length: {LineString(final_route).length:.2f}")
        zrp.plot_solution()
    else:
        print("Failed to find valid route - possible issues:")
        print("1. Obstacles may be too close together")
        print("2. Start point may be in a difficult position")
        print("3. Epsilon values may need adjustment")
        
        # Plot whatever we have
        zrp.plot_solution()