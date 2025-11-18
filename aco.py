import numpy as np
import random
from typing import List, Tuple
import time

class ACO_TSP:
    def __init__(self, dist_matrix: np.ndarray):
        self.dist = dist_matrix
        self.n = len(dist_matrix)
        self.pheromone = np.ones((self.n, self.n)) * 0.1
        self.alpha = 1.0
        self.beta = 5.0
        self.evaporation = 0.6
        self.Q = 200
        self.n_ants = max(80, self.n * 5)
        self.iterations = max(800, self.n * 30)

    def run(self) -> Tuple[List[int], float]:
        best_path = None
        best_cost = float('inf')
        history = []  # best cost per iteration
        history_times = []  # cumulative time per iteration
        start_total = time.perf_counter()

        for it in range(self.iterations):
            all_paths = []
            all_costs = []

            for _ in range(self.n_ants):
                path = [0]
                visited = {0}
                while len(path) < self.n:
                    current = path[-1]
                    attract = (self.pheromone[current] ** self.alpha) * ((1.0 / (self.dist[current] + 1e-10)) ** self.beta)
                    attract[list(visited)] = 0
                    probs = attract / attract.sum() if attract.sum() > 0 else np.ones(self.n) / (self.n - len(visited))
                    next_city = np.random.choice(self.n, p=probs)
                    path.append(next_city)
                    visited.add(next_city)

                path.append(0)
                cost = sum(self.dist[path[i]][path[i+1]] for i in range(self.n))
                all_paths.append(path[:-1])
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # pheromone update
            self.pheromone *= (1 - self.evaporation)
            for path, cost in zip(all_paths, all_costs):
                for i in range(self.n):
                    a, b = path[i], path[(i + 1) % self.n]
                    # avoid division by zero
                    if cost > 0:
                        self.pheromone[a][b] += self.Q / cost
                        self.pheromone[b][a] += self.Q / cost

            history.append(best_cost)
            # record cumulative time since run start
            history_times.append(time.perf_counter() - start_total)

        # build detailed log for the final best path (per-edge costs)
        detailed_log = []
        if best_path is not None:
            for i in range(len(best_path) - 1):
                a = best_path[i]
                b = best_path[i+1]
                c = self.dist[a][b]
                detailed_log.append(f"{a} -> {b}: cost={c:.3f}")

        return best_path, best_cost, history, history_times, detailed_log