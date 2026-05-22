"""
城市物流配送数据生成器
模拟城市中的配送中心（仓库）和多个配送点
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DeliveryPoint:
    """配送点：包含坐标、需求量、时间窗"""
    id: int
    x: float
    y: float
    demand: float  # 货物需求量
    ready_time: float  # 最早服务时间
    due_time: float  # 最晚服务时间


@dataclass
class DeliveryScenario:
    """配送场景：包含仓库和配送点列表"""
    name: str
    depot: Tuple[float, float]  # 仓库坐标（出发点/返回点）
    points: List[DeliveryPoint]  # 配送点列表
    distance_matrix: np.ndarray  # 距离矩阵 (包含仓库，索引0为仓库)


def generate_random_scenario(
    n: int = 20,
    city_size: float = 50.0,
    seed: int = 42,
    depot_center: bool = True,
) -> DeliveryScenario:
    """
    生成随机城市配送场景

    参数:
        n: 配送点数量
        city_size: 城市范围 [0, city_size] × [0, city_size]
        seed: 随机种子
        depot_center: 仓库是否在城市中心

    返回:
        DeliveryScenario 对象
    """
    rng = np.random.default_rng(seed)

    # 生成仓库位置
    if depot_center:
        depot = (city_size / 2, city_size / 2)
    else:
        depot = (rng.uniform(0, city_size), rng.uniform(0, city_size))

    # 生成配送点 - 使用聚类模拟城市区域
    points = []
    n_clusters = max(2, n // 8)
    cluster_centers = rng.uniform(city_size * 0.1, city_size * 0.9, (n_clusters, 2))
    cluster_std = city_size * 0.08

    for i in range(n):
        center = cluster_centers[i % n_clusters]
        x = np.clip(rng.normal(center[0], cluster_std), 0, city_size)
        y = np.clip(rng.normal(center[1], cluster_std), 0, city_size)
        demand = round(rng.uniform(5, 50), 1)
        ready_time = rng.uniform(0, 4)  # 最早服务时间 (小时)
        due_time = ready_time + rng.uniform(2, 8)  # 最晚服务时间

        points.append(DeliveryPoint(
            id=i,
            x=round(x, 2),
            y=round(y, 2),
            demand=demand,
            ready_time=round(ready_time, 2),
            due_time=round(due_time, 2),
        ))

    # 计算距离矩阵（包含仓库为索引0）
    all_coords = [depot] + [(p.x, p.y) for p in points]
    dist_matrix = compute_distance_matrix(all_coords)

    return DeliveryScenario(
        name=f"urban_{n}_points",
        depot=depot,
        points=points,
        distance_matrix=dist_matrix,
    )


def generate_grid_scenario(
    rows: int = 5,
    cols: int = 5,
    spacing: float = 10.0,
    seed: int = 42,
) -> DeliveryScenario:
    """生成网格状配送场景，模拟街区配送"""
    rng = np.random.default_rng(seed)

    depot = (cols * spacing / 2, rows * spacing / 2)

    points = []
    for r in range(rows):
        for c in range(cols):
            # 添加随机偏移模拟实际道路情况
            x = c * spacing + rng.uniform(-2, 2)
            y = r * spacing + rng.uniform(-2, 2)
            points.append(DeliveryPoint(
                id=r * cols + c,
                x=round(x, 2),
                y=round(y, 2),
                demand=round(rng.uniform(5, 30), 1),
                ready_time=0,
                due_time=24,
            ))

    all_coords = [depot] + [(p.x, p.y) for p in points]
    dist_matrix = compute_distance_matrix(all_coords)

    return DeliveryScenario(
        name=f"grid_{rows}x{cols}",
        depot=depot,
        points=points,
        distance_matrix=dist_matrix,
    )


def compute_distance_matrix(
    coords: List[Tuple[float, float]],
    metric: str = "euclidean",
) -> np.ndarray:
    """
    计算距离矩阵

    参数:
        coords: 坐标列表
        metric: 距离度量 ("euclidean" 或 "manhattan")

    返回:
        n×n 距离矩阵
    """
    n = len(coords)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            if metric == "manhattan":
                d = abs(dx) + abs(dy)
            else:
                d = np.sqrt(dx * dx + dy * dy)
            dist[i][j] = d
            dist[j][i] = d

    return dist


if __name__ == "__main__":
    scenario = generate_random_scenario(20)
    print(f"Scenario: {scenario.name}")
    print(f"Depot: {scenario.depot}")
    print(f"Points: {len(scenario.points)}")
    print(f"Distance matrix shape: {scenario.distance_matrix.shape}")
    print(f"Max distance: {scenario.distance_matrix.max():.2f}")
    print(f"Min distance: {scenario.distance_matrix[scenario.distance_matrix > 0].min():.2f}")
