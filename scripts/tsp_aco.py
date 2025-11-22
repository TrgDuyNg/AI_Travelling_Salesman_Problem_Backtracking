"""
Travelling Salesman Problem - Ant Colony Optimization (No Library)
Giải bài toán người du lịch bằng thuật toán tối ưu hóa đàn kiến
"""

import time
import random
from typing import List, Tuple

class TSP_ACO:
    def __init__(self, cities: List[str], distance_matrix,
                 n_ants: int = 20, n_iterations: int = 50,
                 alpha: float = 1.0, beta: float = 2.0,
                 evaporation_rate: float = 0.5, q: float = 100):
        """
        Khởi tạo thuật toán ACO cho TSP
        
        Args:
            cities: Danh sách tên các thành phố
            distance_matrix: Ma trận khoảng cách giữa các thành phố
            n_ants: Số lượng kiến
            n_iterations: Số lần lặp
            alpha: Trọng số pheromone
            beta: Trọng số heuristic (khoảng cách)
            evaporation_rate: Tỷ lệ bay hơi pheromone
            q: Hằng số cập nhật pheromone
        """
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        
    
        self.pheromone = [[1.0 for _ in range(self.n_cities)] for _ in range(self.n_cities)]
        
      
        self.heuristic = [[0.0 for _ in range(self.n_cities)] for _ in range(self.n_cities)]
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j and distance_matrix[i][j] > 0:
                    self.heuristic[i][j] = 1.0 / distance_matrix[i][j]
        
        self.best_route = None
        self.best_distance = float('inf')
        self.execution_time = 0
        self.convergence_data = []
        self.steps_log = []
        
    def calculate_route_distance(self, route: List[int]) -> float:
        """Tính tổng khoảng cách của một tuyến đường"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i + 1]]
       
        total_distance += self.distance_matrix[route[-1]][route[0]]
        return total_distance
    
    def select_next_city(self, current_city: int, unvisited: List[int]) -> int:
        """
        Chọn thành phố tiếp theo dựa trên xác suất
        Xác suất = (pheromone^alpha) * (heuristic^beta)
        """
        probabilities = []
        total_probability = 0
        
        for city in unvisited:
            pheromone_value = (self.pheromone[current_city][city] ** self.alpha)
            heuristic_value = (self.heuristic[current_city][city] ** self.beta)
            probability = pheromone_value * heuristic_value
            probabilities.append(probability)
            total_probability += probability
        
      
        if total_probability == 0:
            return random.choice(unvisited)
        
        probabilities = [p / total_probability for p in probabilities]
        
     
        rand = random.random()
        cumulative = 0
        for i, city in enumerate(unvisited):
            cumulative += probabilities[i]
            if rand <= cumulative:
                return city
        
        return unvisited[-1]
    
    def construct_solution(self) -> Tuple[List[int], float]:
        """Xây dựng một tuyến đường cho một con kiến"""
        start_city = random.randint(0, self.n_cities - 1)
        route = [start_city]
        unvisited = list(range(self.n_cities))
        unvisited.remove(start_city)
        
       
        while unvisited:
            current_city = route[-1]
            next_city = self.select_next_city(current_city, unvisited)
            route.append(next_city)
            unvisited.remove(next_city)
        
        distance = self.calculate_route_distance(route)
        return route, distance
    
    def update_pheromone(self, all_routes: List[Tuple[List[int], float]]):
        """Cập nhật ma trận pheromone"""
       
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.pheromone[i][j] *= (1 - self.evaporation_rate)
        
      
        for route, distance in all_routes:
            pheromone_deposit = self.q / distance
            for i in range(len(route) - 1):
                self.pheromone[route[i]][route[i + 1]] += pheromone_deposit
                self.pheromone[route[i + 1]][route[i]] += pheromone_deposit
            
          
            self.pheromone[route[-1]][route[0]] += pheromone_deposit
            self.pheromone[route[0]][route[-1]] += pheromone_deposit
    
    def solve(self, verbose: bool = False) -> dict:
        """
        Giải bài toán TSP bằng ACO
        
        Args:
            verbose: In chi tiết các bước
            
        Returns:
            dict: Kết quả gồm tuyến đường, khoảng cách, thời gian, log
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"THUẬT TOÁN ACO (ANT COLONY OPTIMIZATION)")
            print(f"{'='*70}")
            print(f"Số thành phố: {self.n_cities}")
            print(f"Danh sách thành phố: {', '.join(self.cities)}")
            print(f"Số kiến: {self.n_ants}")
            print(f"Số iterations: {self.n_iterations}")
            print(f"Tham số Alpha (pheromone): {self.alpha}")
            print(f"Tham số Beta (heuristic): {self.beta}")
            print(f"Tỷ lệ bay hơi: {self.evaporation_rate}")
            print(f"Q constant: {self.q}")
            print(f"{'='*70}\n")
        
        for iteration in range(self.n_iterations):
            all_routes = []
            
          
            for ant in range(self.n_ants):
                route, distance = self.construct_solution()
                all_routes.append((route, distance))
                
               
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route
                    
                    if verbose and len(self.steps_log) < 20:
                        log_msg = f"Iteration {iteration + 1}: Tìm tuyến đường tốt hơn: {self.best_distance:.2f} km"
                        self.steps_log.append(log_msg)
            
         
            self.update_pheromone(all_routes)
            
        
            self.convergence_data.append(self.best_distance)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: "
                      f"Khoảng cách tốt nhất = {self.best_distance:.2f} km")
        
        self.execution_time = time.time() - start_time
        
        
        best_route_names = [self.cities[i] for i in self.best_route]
        
        if verbose:
            print(f"\nKết quả:")
            print(f"Tuyến đường tốt nhất: {' -> '.join(best_route_names)} -> {best_route_names[0]}")
            print(f"Tổng khoảng cách: {self.best_distance:.2f} km")
            print(f"Thời gian thực thi: {self.execution_time:.4f} giây")
            print(f"{'='*70}\n")
        
        return {
            'route': best_route_names,
            'distance': self.best_distance,
            'time': self.execution_time,
            'algorithm': 'ACO (Ant Colony Optimization)',
            'convergence': self.convergence_data,
            'steps': self.steps_log,
            'parameters': {
                'n_ants': self.n_ants,
                'n_iterations': self.n_iterations,
                'alpha': self.alpha,
                'beta': self.beta,
                'evaporation_rate': self.evaporation_rate,
                'q': self.q
            }
        }