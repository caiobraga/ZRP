import random
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

class ZRPAlgorithm:
    def __init__(self, container_polygon, obstacle_polygons, epsilon2=0.1, epsilon3=0.1):
        self.Π = container_polygon
        self.P = obstacle_polygons
        self.ε2 = epsilon2
        self.ε3 = epsilon3
        self.route = None
        self.length_history = []
        self.zrp_pairs = self.calculate_zrp_pairs()
        
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
        """Compute path that avoids all given obstacles"""
        direct_line = LineString([p1, p2])
        
        # Check if direct path is valid
        valid = True
        for obs in obstacles:
            if direct_line.crosses(obs):
                valid = False
                break
        if valid:
            return [p1, p2]
        
        # If not, find path around obstacles
        path = [p1]
        remaining_obstacles = obstacles.copy()
        
        while remaining_obstacles:
            # Find next obstacle in path
            current_line = LineString([path[-1], p2])
            next_obs = None
            for obs in remaining_obstacles:
                if current_line.crosses(obs):
                    next_obs = obs
                    break
            
            if not next_obs:
                break
                
            # Find way around this obstacle
            hull_points = list(next_obs.exterior.coords)
            
            # Find tangent points
            t1 = self.find_tangent(path[-1], next_obs)
            t2 = self.find_tangent(p2, next_obs)
            
            idx1 = min(range(len(hull_points)),
                      key=lambda i: np.linalg.norm(np.array(hull_points[i]) - np.array(t1)))
            idx2 = min(range(len(hull_points)),
                      key=lambda i: np.linalg.norm(np.array(hull_points[i]) - np.array(t2)))
            
            # Add path around obstacle
            if idx1 < idx2:
                path.extend(hull_points[idx1:idx2+1])
            else:
                path.extend(hull_points[idx2:idx1+1][::-1])
            
            remaining_obstacles.remove(next_obs)
        
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
    
    def validate_route(self):
        """Check if route is valid (doesn't cross any obstacles)"""
        if not self.route or len(self.route) < 2:
            return False
            
        route_line = LineString(self.route)
        for i, poly in enumerate(self.P):
            if route_line.crosses(poly):
                print(f"Invalid route - crosses obstacle {i}")
                return False
        return True
    
    def repair_route(self):
        """Attempt to repair invalid route segments"""
        if not self.route:
            return False
            
        new_route = [self.route[0]]
        for i in range(1, len(self.route)):
            segment = LineString([new_route[-1], self.route[i]])
            
            # Check if segment crosses any obstacle
            needs_repair = False
            for poly in self.P:
                if segment.crosses(poly):
                    needs_repair = True
                    break
            
            if needs_repair:
                # Recalculate this segment with obstacle avoidance
                repaired_segment = self.extended_shortest_path(new_route[-1], self.route[i], self.P)
                if len(repaired_segment) > 1:
                    new_route.extend(repaired_segment[1:])
            else:
                new_route.append(self.route[i])
        
        self.route = new_route
        return self.validate_route()
    
    def algorithm5(self, max_iterations=100, improvement_threshold=0.5):
        """Main optimization algorithm with validation"""
        # Build initial route
        self.build_initial_route()
        
        # Validate and repair initial route
        if not self.validate_route():
            print("Initial route invalid - attempting repair...")
            if not self.repair_route():
                print("Failed to repair initial route")
                return None
        
        # Optimize
        L1 = LineString(self.route).length
        self.length_history = [L1]
        iteration = 0
        improvement = float('inf')
        
        while iteration < max_iterations and improvement > improvement_threshold:
            new_route = self.route.copy()
            made_improvement = False
            
            # Optimize each triplet
            k = len(self.P)
            for i in range(k):
                P1 = self.P[(i-1) % k]
                P2 = self.P[i]
                P3 = self.P[(i+1) % k]
                
                refined = self.procedure4(P1, P2, P3, 
                                        *self.zrp_pairs[(i-1)%k], 
                                        *self.zrp_pairs[i], 
                                        *self.zrp_pairs[(i+1)%k])
                
                if refined and self.validate_segment(refined):
                    try:
                        # Find and replace segment
                        start_idx, end_idx = self.find_segment_indices(new_route, refined)
                        if start_idx is not None and end_idx is not None:
                            new_route = new_route[:start_idx] + refined + new_route[end_idx+1:]
                            made_improvement = True
                    except Exception as e:
                        print(f"Segment replacement error: {e}")
                        continue
            
            # Validate and update
            new_length = LineString(new_route).length
            improvement = L1 - new_length
            
            if improvement > improvement_threshold and self.validate_route(new_route):
                self.route = new_route
                L1 = new_length
                self.length_history.append(L1)
                print(f"Iteration {iteration+1}: Improved to {L1:.2f}")
            else:
                print(f"Iteration {iteration+1}: No improvement")
            
            iteration += 1
        
        # Final validation
        if not self.validate_route():
            print("Final route invalid - attempting repair...")
            if not self.repair_route():
                print("Failed to repair final route")
                return None
                
        return self.route
    
    def build_initial_route(self):
        """Build initial route connecting all ZRP pairs"""
        self.route = []
        for i, (p1, p2) in enumerate(self.zrp_pairs):
            self.route.append(p1)
            if i < len(self.zrp_pairs)-1:
                next_p1 = self.zrp_pairs[i+1][0]
                path_segment = self.extended_shortest_path(p2, next_p1, 
                                                         [p for p in self.P if p not in [self.P[i], self.P[i+1]]])
                self.route.extend(path_segment[1:])
        
        # Close the loop
        if len(self.zrp_pairs) > 1:
            last_p2 = self.zrp_pairs[-1][1]
            first_p1 = self.zrp_pairs[0][0]
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
    
    def validate_route(self, route=None):
        """Check if route is valid (doesn't cross obstacles)"""
        route_to_check = route if route is not None else self.route
        if not route_to_check or len(route_to_check) < 2:
            return False
            
        route_line = LineString(route_to_check)
        for i, poly in enumerate(self.P):
            if route_line.crosses(poly):
                print(f"Route crosses obstacle {i}")
                return False
        return True
    
    def plot_solution(self):
        """Visualize the solution with validation"""
        plt.figure(figsize=(12, 8))
        
        # Plot container
        x, y = self.Π.exterior.xy
        plt.plot(x, y, 'b-', linewidth=3, label='Container')
        plt.fill(x, y, 'b', alpha=0.1)
        
        # Plot obstacles with labels
        for i, poly in enumerate(self.P):
            x, y = poly.exterior.xy
            plt.fill(x, y, 'r', alpha=0.3)
            plt.plot(x, y, 'r-', linewidth=2)
            plt.text(poly.centroid.x, poly.centroid.y, f'Obstacle {i}', 
                    ha='center', va='center', color='darkred')
        
        # Plot ZRP pairs
        for i, (p1, p2) in enumerate(self.zrp_pairs):
            plt.plot(p1[0], p1[1], 'go', markersize=10, label='Entry' if i == 0 else "")
            plt.plot(p2[0], p2[1], 'yo', markersize=10, label='Exit' if i == 0 else "")
            plt.text(p1[0], p1[1], f'E{i}', ha='right', va='bottom', fontsize=8)
            plt.text(p2[0], p2[1], f'X{i}', ha='left', va='bottom', fontsize=8)
        
        # Plot route if valid
        if self.route and len(self.route) > 1:
            if self.validate_route():
                route_x, route_y = zip(*self.route)
                plt.plot(route_x, route_y, 'g-', linewidth=2, label='Route')
                plt.plot(route_x[0], route_y[0], 'go', markersize=12, 
                        markeredgecolor='k', label='Start/End')
                
                # Add direction arrows
                for i in range(0, len(self.route)-1, max(1, len(self.route)//10)):
                    dx = self.route[i+1][0] - self.route[i][0]
                    dy = self.route[i+1][1] - self.route[i][1]
                    plt.arrow(self.route[i][0], self.route[i][1], 
                             dx*0.8, dy*0.8, head_width=0.2, 
                             head_length=0.3, fc='green', ec='green')
                plt.title('Valid Route - Length: {:.2f}'.format(LineString(self.route).length))
            else:
                plt.title('INVALID ROUTE - Crosses Obstacles', color='red')
        
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
    # Create interesting environment
    container, obstacles = create_interesting_environment()
    
    # Create and run algorithm
    zrp = ZRPAlgorithm(container, obstacles, epsilon2=0.05, epsilon3=0.05)
    
    # Print calculated entry/exit points
    print("Entry/Exit points for each obstacle:")
    for i, (entry, exit_pt) in enumerate(zrp.zrp_pairs):
        print(f"Obstacle {i}:")
        print(f"  Entry: {entry}")
        print(f"  Exit: {exit_pt}")
        print(f"  All vertices: {list(zrp.P[i].exterior.coords)[:-1]}")
    
    # Solve with validation
    final_route = zrp.algorithm5(max_iterations=30, improvement_threshold=0.5)
    
    if final_route:
        print("\nOptimization completed successfully")
        print(f"Final route length: {LineString(final_route).length:.2f}")
        
        # Validate final route
        if zrp.validate_route():
            print("Route is valid - no obstacles crossed")
        else:
            print("Warning: Route crosses obstacles")
    else:
        print("\nOptimization failed - no valid route found")
    
    # Visualize
    zrp.plot_solution()
    
    # Plot all potential entry/exit points
    plt.figure(figsize=(12, 8))
    
    # Plot container
    x, y = container.exterior.xy
    plt.plot(x, y, 'b-', linewidth=3, label='Container')
    plt.fill(x, y, 'b', alpha=0.1)
    
    # Plot obstacles with all candidate points
    for i, poly in enumerate(obstacles):
        x, y = poly.exterior.xy
        plt.fill(x, y, 'r', alpha=0.3)
        plt.plot(x, y, 'r-', linewidth=2)
        plt.text(poly.centroid.x, poly.centroid.y, f'Obstacle {i}', 
                ha='center', va='center', color='darkred')
        
        # Plot all vertices and edge midpoints
        vertices = list(poly.exterior.coords)[:-1]
        edges = list(zip(vertices, vertices[1:] + [vertices[0]]))
        edge_midpoints = [((e[0][0]+e[1][0])/2, (e[0][1]+e[1][1])/2) for e in edges]
        
        # Plot vertices
        vx, vy = zip(*vertices)
        plt.plot(vx, vy, 'bo', markersize=6, label='Vertices' if i == 0 else "")
        
        # Plot edge midpoints
        mx, my = zip(*edge_midpoints)
        plt.plot(mx, my, 'yo', markersize=6, label='Edge Midpoints' if i == 0 else "")
        
        # Highlight selected entry/exit points
        entry, exit_pt = zrp.zrp_pairs[i]
        plt.plot(entry[0], entry[1], 'go', markersize=10, label='Selected Entry' if i == 0 else "")
        plt.plot(exit_pt[0], exit_pt[1], 'mo', markersize=10, label='Selected Exit' if i == 0 else "")
    
    # Plot route if exists
    if zrp.route and len(zrp.route) > 1:
        route_x, route_y = zip(*zrp.route)
        plt.plot(route_x, route_y, 'g-', linewidth=2, label='Route')
        plt.plot(route_x[0], route_y[0], 'go', markersize=12, 
                markeredgecolor='k', label='Start/End')
    
    plt.legend(loc='upper right')
    plt.title('All Potential Entry/Exit Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()