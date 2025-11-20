"""
Travelling Salesman Problem - Backtracking Solution (No Library)
Giải bài toán người du lịch bằng thuật toán quay lui thuần túy
"""

import time
from typing import List, Tuple

class TSPBacktracking:
    def __init__(self, cities: List[str], distance_matrix):
        """
        Khởi tạo bài toán TSP với Backtracking
        
        Args:
            cities: Danh sách tên các thành phố
            distance_matrix: Ma trận khoảng cách giữa các thành phố
        """
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.n_cities = len(cities)
        self.best_route = None
        self.best_distance = float('inf')
        self.execution_time = 0
        self.steps_log = []
        self.explored_routes = 0
        
    def calculate_route_distance(self, route: List[int]) -> float:
        """Tính tổng khoảng cách của một tuyến đường"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i + 1]]
        # Quay về thành phố xuất phát
        total_distance += self.distance_matrix[route[-1]][route[0]]
        return total_distance
    
    def backtrack(self, current_route: List[int], unvisited: set, current_distance: float):
        """
        Hàm backtracking - khám phá tất cả các tuyến đường có thể
        
        Args:
            current_route: Tuyến đường hiện tại (danh sách index thành phố)
            unvisited: Tập hợp các thành phố chưa được thăm
            current_distance: Khoảng cách tích lũy từ đầu
        """
        self.explored_routes += 1
        
        # Nếu đã thăm hết tất cả các thành phố
        if len(unvisited) == 0:
            # Tính khoảng cách để quay về thành phố xuất phát
            final_distance = current_distance + self.distance_matrix[current_route[-1]][current_route[0]]
            
            if final_distance < self.best_distance:
                self.best_distance = final_distance
                self.best_route = current_route[:]
                log_msg = f"Tìm tuyến đường tốt hơn: {self.best_distance:.2f}"
                self.steps_log.append(log_msg)
            return
        
        # Pruning: nếu khoảng cách hiện tại đã vượt quá best_distance, bỏ qua nhánh này
        if current_distance >= self.best_distance:
            return
        
        # Thử tất cả các thành phố chưa thăm
        for next_city in list(unvisited):
            distance_to_next = self.distance_matrix[current_route[-1]][next_city]
            
            # Ghi lại bước
            if len(self.steps_log) < 50:  # Giới hạn log để không quá dài
                log_msg = f"→ Đi từ {self.cities[current_route[-1]]} sang {self.cities[next_city]} "
                log_msg += f"(khoảng cách: {distance_to_next:.2f}, tích lũy: {current_distance + distance_to_next:.2f})"
                self.steps_log.append(log_msg)
            
            # Thêm thành phố vào tuyến đường
            current_route.append(next_city)
            unvisited.remove(next_city)
            
            # Gọi đệ quy
            self.backtrack(current_route, unvisited, current_distance + distance_to_next)
            
            # Quay lui (backtrack)
            current_route.pop()
            unvisited.add(next_city)
    
    def solve(self, verbose: bool = False) -> dict:
        """
        Giải bài toán TSP bằng Backtracking
        
        Args:
            verbose: In chi tiết các bước
            
        Returns:
            dict: Kết quả gồm tuyến đường, khoảng cách, thời gian, log
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"THUẬT TOÁN BACKTRACKING - GIẢI BÀI TOÁN NGƯỜI DU LỊCH")
            print(f"{'='*70}")
            print(f"Số thành phố: {self.n_cities}")
            print(f"Danh sách thành phố: {', '.join(self.cities)}")
            print(f"Phương pháp: Quay lui (Backtracking)")
            print(f"Độ phức tạp: O(n!) - Tất cả các hoán vị")
            print(f"{'='*70}\n")
        
        # Bắt đầu từ thành phố 0
        initial_route = [0]
        unvisited = set(range(1, self.n_cities))
        
        self.backtrack(initial_route, unvisited, 0)
        
        self.execution_time = time.time() - start_time
        
        # Chuyển đổi route từ index sang tên thành phố
        best_route_names = [self.cities[i] for i in self.best_route]
        
        if verbose:
            print(f"\nKếT QUẢ:")
            print(f"Tuyến đường tốt nhất: {' -> '.join(best_route_names)} -> {best_route_names[0]}")
            print(f"Tổng khoảng cách: {self.best_distance:.2f} km")
            print(f"Thời gian thực thi: {self.execution_time:.4f} giây")
            print(f"Số tuyến đường khám phá: {self.explored_routes}")
            print(f"{'='*70}\n")
        
        return {
            'route': best_route_names,
            'distance': self.best_distance,
            'time': self.execution_time,
            'algorithm': 'Backtracking (Quay lui)',
            'explored_routes': self.explored_routes,
            'steps': self.steps_log
        }