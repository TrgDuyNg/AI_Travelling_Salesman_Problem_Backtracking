import time
import numpy as np
from typing import List
from utils import compute_mst_cost

class TSPBacktracking:
    def __init__(self, coords: np.ndarray, names: List[str]):
        self.coords = coords
        self.n = len(coords)
        self.names = names
        self.dist = np.linalg.norm(coords[:, None] - coords, axis=-1)

        self.best_cost = float('inf')
        self.best_path = []
        self.current_path = []
        self.visited = 0
        self.current_cost = 0.0
        self.node_count = 0
        self.log_lines = []

        self._init_upper_bound()

    def _init_upper_bound(self):
        path, cost = self._nearest_neighbor()
        self.best_path = path
        self.best_cost = cost

    def _nearest_neighbor(self):
        visited = [False] * self.n
        path = [0]
        visited[0] = True
        cost = 0.0
        curr = 0
        for _ in range(self.n - 1):
            next_city = min((j for j in range(self.n) if not visited[j]), key=lambda j: self.dist[curr][j])
            cost += self.dist[curr][next_city]
            path.append(next_city)
            visited[next_city] = True
            curr = next_city
        cost += self.dist[curr][0]
        path.append(0)
        return path, cost

    def solve(self) -> tuple:
        self.log_lines.clear()
        self.node_count = 0
        start = time.time()

        self.current_path = [0]
        self.visited = 1 << 0
        self.current_cost = 0.0

        self._backtrack(1)

        elapsed = time.time() - start
        return self.best_path, self.best_cost, elapsed, self.log_lines, self.node_count

    def _backtrack(self, depth: int):
        self.node_count += 1
        curr = self.current_path[-1]

        if depth == self.n:
            return_cost = self.current_cost + self.dist[curr][0]
            if return_cost < self.best_cost:
                self.best_cost = return_cost
                self.best_path = self.current_path + [0]
            return

        unvisited = [i for i in range(self.n) if not (self.visited & (1 << i))]
        if unvisited:
            remaining = list(set(unvisited + [curr, 0]))
            mst_cost = compute_mst_cost(self.dist, remaining)
            lb = self.current_cost + mst_cost
            if lb >= self.best_cost:
                return

        candidates = sorted(((self.dist[curr][j], j) for j in range(self.n) if not (self.visited & (1 << j))))
        for add_cost, next_city in candidates:
            new_cost = self.current_cost + add_cost
            if new_cost + self.dist[next_city][0] >= self.best_cost:
                continue

            self.current_path.append(next_city)
            self.visited |= (1 << next_city)
            self.current_cost = new_cost
            self._backtrack(depth + 1)
            self.current_path.pop()
            self.visited ^= (1 << next_city)
            self.current_cost -= add_cost