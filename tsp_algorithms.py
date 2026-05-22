"""
TSP 近似算法实现
包含: 精确DP, 最近邻, Christofides (3/2近似), 2-Opt, 模拟退火
"""

import numpy as np
import time
from typing import List, Tuple, Callable
from itertools import combinations


def tour_length(tour: List[int], dist: np.ndarray) -> float:
    """计算路径总长度"""
    total = 0.0
    for i in range(len(tour)):
        total += dist[tour[i]][tour[(i + 1) % len(tour)]]
    return total


def solve_tsp_dp(dist: np.ndarray) -> Tuple[List[int], float]:
    """
    Held-Karp 动态规划求解 TSP（精确解）
    时间复杂度 O(n²·2ⁿ)，仅适用于 n ≤ 15

    参数:
        dist: 距离矩阵 (索引0为起点/仓库)

    返回:
        (最优路径, 最优长度)
    """
    n = dist.shape[0]
    if n > 15:
        raise ValueError(f"DP求解器仅支持 n≤15，当前 n={n}")

    # dp[mask][i] = 从0出发，经过mask中所有节点，最后到达i的最短距离
    INF = float('inf')
    dp = np.full((1 << n, n), INF)
    parent = np.full((1 << n, n), -1, dtype=int)

    dp[1][0] = 0  # 起点

    for mask in range(1 << n):
        if not (mask & 1):  # mask必须包含起点
            continue
        for last in range(n):
            if not (mask >> last & 1) or dp[mask][last] == INF:
                continue
            for nxt in range(n):
                if mask >> nxt & 1:
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = dp[mask][last] + dist[last][nxt]
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last

    # 回到起点
    full_mask = (1 << n) - 1
    best = INF
    best_last = -1
    for last in range(1, n):
        cost = dp[full_mask][last] + dist[last][0]
        if cost < best:
            best = cost
            best_last = last

    # 回溯路径
    tour = [0]
    mask = full_mask
    last = best_last
    while last != 0:
        tour.append(last)
        prev = parent[mask][last]
        mask ^= (1 << last)
        last = prev
    tour.append(0)

    return tour, best


def solve_tsp_nearest_neighbor(dist: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    最近邻贪心算法
    时间复杂度 O(n²)，简单直观，解的质量一般

    参数:
        dist: 距离矩阵
        start: 起始节点索引

    返回:
        (路径, 长度)
    """
    n = dist.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True

    current = start
    for _ in range(n - 1):
        # 寻找最近的未访问节点
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist[current][j] < best_dist:
                best_dist = dist[current][j]
                best_next = j

        tour.append(best_next)
        visited[best_next] = True
        current = best_next

    tour.append(start)  # 回到起点
    total = tour_length(tour, dist)
    return tour, total


def solve_tsp_nearest_insertion(dist: np.ndarray) -> Tuple[List[int], float]:
    """
    最近插入法构造 TSP 回路
    以仓库为起点，每次选择离当前回路最近的节点插入最优位置
    时间复杂度 O(n³)
    """
    n = dist.shape[0]
    unvisited = set(range(1, n))
    tour = [0, 0]  # 初始回路

    while unvisited:
        # 找离当前回路最近的未访问节点
        best_node = -1
        best_dist = float('inf')
        for v in unvisited:
            for u in tour[:-1]:  # 回路中的节点（不重复算0）
                if dist[v][u] < best_dist:
                    best_dist = dist[v][u]
                    best_node = v

        # 找到最佳插入位置（最小增量）
        best_pos = -1
        best_increase = float('inf')
        for i in range(len(tour) - 1):
            increase = (dist[tour[i]][best_node] +
                        dist[best_node][tour[i + 1]] -
                        dist[tour[i]][tour[i + 1]])
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)

    total = tour_length(tour, dist)
    return tour, total


def solve_tsp_christofides(dist: np.ndarray) -> Tuple[List[int], float]:
    """
    Christofides 算法 —— 3/2 近似比保证（度量 TSP）
    步骤:
      1. 构造最小生成树 (MST)
      2. 找出 MST 中奇度顶点
      3. 在奇度顶点上求最小权完美匹配
      4. 合并 MST + 匹配 → 欧拉图
      5. 求欧拉回路
      6. 短路得到哈密顿回路
    时间复杂度 O(n³)
    """
    n = dist.shape[0]
    if n <= 2:
        return [0, 0] if n == 1 else [0, 1, 0], dist[0][1] * 2

    # Step 1: Prim 算法求 MST（邻接表形式）
    mst_adj = [[] for _ in range(n)]
    in_mst = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0
    parent = [-1] * n

    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: min_edge[i])
        in_mst[u] = True
        if parent[u] != -1:
            mst_adj[u].append(parent[u])
            mst_adj[parent[u]].append(u)
        for v in range(n):
            if not in_mst[v] and dist[u][v] < min_edge[v]:
                min_edge[v] = dist[u][v]
                parent[v] = u

    # Step 2: 找奇度顶点
    odd_vertices = [i for i in range(n) if len(mst_adj[i]) % 2 == 1]

    # Step 3: 贪心最小权完美匹配
    matched = [False] * n
    matching_edges = []
    odd_set = set(odd_vertices)

    while odd_set:
        u = min(odd_set, key=lambda x: x)  # 取一个奇度顶点
        best_v = -1
        best_d = float('inf')
        for v in odd_set:
            if v != u and dist[u][v] < best_d:
                best_d = dist[u][v]
                best_v = v
        matching_edges.append((u, best_v))
        odd_set.remove(u)
        odd_set.remove(best_v)

    # Step 4: 合并 MST + 匹配 → 欧拉多图
    euler_adj = [list(neighbors) for neighbors in mst_adj]
    for u, v in matching_edges:
        euler_adj[u].append(v)
        euler_adj[v].append(u)

    # Step 5: Hierholzer 算法求欧拉回路
    edge_count = {}
    for u in range(n):
        for v in euler_adj[u]:
            key = (min(u, v), max(u, v))
            edge_count[key] = edge_count.get(key, 0) + 1

    # 每条边存两次（双向），实际边数为一半
    for key in edge_count:
        edge_count[key] //= 2

    # 使用栈的非递归 Hierholzer
    euler_circuit = []
    stack = [0]
    # 邻接表副本用于删除已使用的边
    adj_copy = [list(neighbors) for neighbors in euler_adj]
    # 用 dict 追踪边剩余次数
    remaining = dict(edge_count)

    while stack:
        u = stack[-1]
        if adj_copy[u]:
            v = adj_copy[u].pop()
            key = (min(u, v), max(u, v))
            if remaining.get(key, 0) > 0:
                remaining[key] -= 1
                stack.append(v)
        else:
            euler_circuit.append(stack.pop())

    # Step 6: 短路得到哈密顿回路
    visited = [False] * n
    tour = []
    for v in euler_circuit:
        if not visited[v]:
            tour.append(v)
            visited[v] = True
    tour.append(tour[0])  # 回到起点

    total = tour_length(tour, dist)
    return tour, total


def two_opt_swap(tour: List[int], i: int, j: int) -> List[int]:
    """执行 2-opt 交换：反转 tour[i+1..j]"""
    new_tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
    return new_tour


def solve_tsp_two_opt(
    dist: np.ndarray,
    init_tour: List[int] = None,
    max_iter: int = 1000,
) -> Tuple[List[int], float]:
    """
    2-Opt 局部搜索优化
    时间复杂度 O(n²) 每次迭代，通常作为其他算法的后处理

    参数:
        dist: 距离矩阵
        init_tour: 初始路径 (None则使用最近邻构造)
        max_iter: 最大迭代次数（无改进的轮数）

    返回:
        (优化后路径, 长度)
    """
    if init_tour is None:
        init_tour, _ = solve_tsp_nearest_neighbor(dist)

    tour = init_tour.copy()
    n = len(tour) - 1  # 不包含回到起点的重复

    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        best_delta = 0
        best_i = best_j = -1

        for i in range(n - 1):
            for j in range(i + 2, n):
                # 检查 2-opt 交换的增量
                old_edges = (dist[tour[i]][tour[i + 1]] +
                             dist[tour[j]][tour[(j + 1) % n]])
                new_edges = (dist[tour[i]][tour[j]] +
                             dist[tour[i + 1]][tour[(j + 1) % n]])
                delta = new_edges - old_edges

                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_delta < 0:
            tour = two_opt_swap(tour, best_i, best_j)
            improved = True

    # 确保回到起点
    if tour[-1] != tour[0]:
        tour.append(tour[0])

    total = tour_length(tour, dist)
    return tour, total


def solve_tsp_simulated_annealing(
    dist: np.ndarray,
    init_temp: float = 1000.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
    iterations_per_temp: int = 100,
    seed: int = 42,
) -> Tuple[List[int], float]:
    """
    模拟退火求解 TSP
    使用 2-opt 邻域 + Metropolis 接受准则
    时间复杂度由冷却调度决定

    参数:
        dist: 距离矩阵
        init_temp: 初始温度
        cooling_rate: 冷却系数
        min_temp: 终止温度
        iterations_per_temp: 每个温度下的迭代次数
        seed: 随机种子

    返回:
        (最佳路径, 最佳长度)
    """
    rng = np.random.default_rng(seed)
    n = dist.shape[0]

    # 用最近邻构造初始解
    current_tour, current_length = solve_tsp_nearest_neighbor(dist)
    best_tour = current_tour.copy()
    best_length = current_length

    temp = init_temp

    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # 随机 2-opt 交换
            n_cities = n
            i = rng.integers(0, n_cities - 2)
            j = rng.integers(i + 2, n_cities)

            old_len = (dist[current_tour[i]][current_tour[i + 1]] +
                       dist[current_tour[j]][current_tour[(j + 1) % n_cities]])
            new_len = (dist[current_tour[i]][current_tour[j]] +
                       dist[current_tour[i + 1]][current_tour[(j + 1) % n_cities]])

            delta = new_len - old_len

            if delta < 0 or rng.random() < np.exp(-delta / temp):
                current_tour = two_opt_swap(current_tour, i, j)
                current_length += delta

                if current_length < best_length:
                    best_tour = current_tour.copy()
                    best_length = current_length

        temp *= cooling_rate

    return best_tour, best_length


class TSPSolver:
    """TSP 求解器统一接口"""

    def __init__(self, dist: np.ndarray):
        self.dist = dist
        self.n = dist.shape[0]

    def solve(self, method: str, **kwargs) -> dict:
        """
        求解 TSP

        参数:
            method: 算法名称
                - "dp" (精确DP)
                - "nn" (最近邻)
                - "ni" (最近插入)
                - "christofides" (Christofides 3/2近似)
                - "nn+2opt" (最近邻 + 2-Opt)
                - "christofides+2opt" (Christofides + 2-Opt)
                - "sa" (模拟退火)
            **kwargs: 传递给具体算法的参数

        返回:
            dict: {"tour": 路径, "length": 长度, "time": 运行时间(秒)}
        """
        start_time = time.perf_counter()

        if method == "dp":
            if self.n > 15:
                return {"tour": None, "length": float('inf'),
                        "time": 0, "error": "n>15, DP不可用"}
            tour, length = solve_tsp_dp(self.dist)

        elif method == "nn":
            tour, length = solve_tsp_nearest_neighbor(self.dist)

        elif method == "ni":
            tour, length = solve_tsp_nearest_insertion(self.dist)

        elif method == "christofides":
            tour, length = solve_tsp_christofides(self.dist)

        elif method == "nn+2opt":
            init_tour, _ = solve_tsp_nearest_neighbor(self.dist)
            tour, length = solve_tsp_two_opt(self.dist, init_tour)

        elif method == "christofides+2opt":
            init_tour, _ = solve_tsp_christofides(self.dist)
            tour, length = solve_tsp_two_opt(self.dist, init_tour)

        elif method == "sa":
            init_temp = kwargs.get("init_temp", 1000.0)
            cooling_rate = kwargs.get("cooling_rate", 0.995)
            tour, length = solve_tsp_simulated_annealing(
                self.dist,
                init_temp=init_temp,
                cooling_rate=cooling_rate,
            )

        else:
            raise ValueError(f"未知算法: {method}")

        elapsed = time.perf_counter() - start_time
        return {"tour": tour, "length": length, "time": elapsed}


def compare_algorithms(
    dist: np.ndarray,
    methods: List[str] = None,
) -> List[dict]:
    """
    比较多种算法在同一实例上的表现

    参数:
        dist: 距离矩阵
        methods: 算法列表

    返回:
        结果列表
    """
    if methods is None:
        methods = ["nn", "ni", "christofides", "nn+2opt", "christofides+2opt", "sa"]

    solver = TSPSolver(dist)
    results = []

    # 如果 n 较小，先用 DP 求最优解
    opt_length = None
    if dist.shape[0] <= 12:
        opt_result = solver.solve("dp")
        opt_length = opt_result["length"]
        results.append({"method": "Optimal (DP)", **opt_result})

    for method in methods:
        result = solver.solve(method)
        result["method"] = method

        if opt_length and opt_length > 0:
            result["gap"] = (result["length"] - opt_length) / opt_length * 100
        else:
            result["gap"] = None

        results.append(result)

    return results


if __name__ == "__main__":
    from data_generator import generate_random_scenario

    # 快速测试
    for n in [8, 15, 20]:
        scenario = generate_random_scenario(n, seed=n)
        print(f"\n{'='*50}")
        print(f"n={n} 配送点")
        print(f"{'='*50}")

        results = compare_algorithms(scenario.distance_matrix)
        for r in results:
            gap_str = f"gap={r['gap']:.2f}%" if r.get('gap') is not None else ""
            print(f"  {r['method']:20s}: length={r['length']:.2f}, time={r['time']:.4f}s {gap_str}")
