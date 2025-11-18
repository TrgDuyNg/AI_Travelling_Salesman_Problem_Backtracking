import numpy as np

def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    return np.linalg.norm(cities[:, None] - cities, axis=-1)

def compute_mst_cost(dist: np.ndarray, cities_list: list) -> float:
    n = len(cities_list)
    if n < 2: return 0.0
    key = [float('inf')] * n
    in_mst = [False] * n
    key[0] = 0
    total = 0.0
    for _ in range(n):
        u = min(range(n), key=lambda i: key[i] if not in_mst[i] else float('inf'))
        in_mst[u] = True
        total += key[u]
        for v in range(n):
            if not in_mst[v] and dist[cities_list[u]][cities_list[v]] < key[v]:
                key[v] = dist[cities_list[u]][cities_list[v]]
    return total