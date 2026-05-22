"""
可视化模块
绘制配送路线图、算法对比图
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import List, Dict, Optional
from data_generator import DeliveryScenario, DeliveryPoint

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_route(
    scenario: DeliveryScenario,
    tour: List[int],
    title: str = "配送路线",
    save_path: Optional[str] = None,
    show_length: bool = True,
):
    """
    绘制单条配送路线

    参数:
        scenario: 配送场景
        tour: 路径（节点索引列表，0为仓库）
        title: 图表标题
        save_path: 保存路径
        show_length: 是否显示总长度
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 提取坐标
    all_coords = [scenario.depot] + [(p.x, p.y) for p in scenario.points]
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]

    # 绘制路径
    route_xs = [all_coords[i][0] for i in tour]
    route_ys = [all_coords[i][1] for i in tour]

    ax.plot(route_xs, route_ys, 'b-', linewidth=1.5, alpha=0.7, zorder=1)
    ax.scatter(route_xs[1:-1], route_ys[1:-1], c='steelblue', s=60,
               zorder=3, edgecolors='white', linewidth=0.5)

    # 标注仓库
    ax.scatter([scenario.depot[0]], [scenario.depot[1]],
               c='red', s=200, marker='s', zorder=5,
               edgecolors='darkred', linewidth=1.5, label='配送中心')

    # 标注起点/终点箭头
    ax.annotate('起点/终点', xy=scenario.depot, xytext=(scenario.depot[0] + 3, scenario.depot[1] + 3),
                fontsize=9, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    # 标注配送点编号
    for p in scenario.points:
        ax.annotate(str(p.id), (p.x, p.y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color='gray')

    # 添加城市区域背景
    ax.set_xlim(min(xs) - 5, max(xs) + 5)
    ax.set_ylim(min(ys) - 5, max(ys) + 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X 坐标 (km)')
    ax.set_ylabel('Y 坐标 (km)')

    total_len = 0
    for i in range(len(tour) - 1):
        total_len += scenario.distance_matrix[tour[i]][tour[i + 1]]

    subtitle = title
    if show_length:
        subtitle += f"\n总里程: {total_len:.2f} km | 配送点: {len(scenario.points)}"

    ax.set_title(subtitle, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_routes(
    scenario: DeliveryScenario,
    results: List[dict],
    save_path: Optional[str] = None,
):
    """
    并排对比多种算法路线

    参数:
        scenario: 配送场景
        results: 算法结果列表，每项含 {"method", "tour", "length"}
        save_path: 保存路径
    """
    valid_results = [r for r in results if r.get("tour") is not None]
    n_methods = len(valid_results)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    all_coords = [scenario.depot] + [(p.x, p.y) for p in scenario.points]

    for idx, result in enumerate(valid_results):
        ax = axes[idx]
        tour = result["tour"]

        route_xs = [all_coords[i][0] for i in tour]
        route_ys = [all_coords[i][1] for i in tour]

        ax.plot(route_xs, route_ys, 'b-', linewidth=1.2, alpha=0.6)
        ax.scatter(route_xs[1:-1], route_ys[1:-1], c='steelblue', s=30)
        ax.scatter([scenario.depot[0]], [scenario.depot[1]],
                   c='red', s=100, marker='s', edgecolors='darkred', linewidth=1)

        ax.set_title(f"{result['method']}\n{result['length']:.2f} km", fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"算法路线对比 — {scenario.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_quality_comparison(
    results: List[dict],
    title: str = "算法解质量对比",
    save_path: Optional[str] = None,
    opt_length: Optional[float] = None,
):
    """
    柱状图对比算法解质量

    参数:
        results: 算法结果列表
        title: 标题
        save_path: 保存路径
        opt_length: 最优解长度（用于标注）
    """
    methods = [r["method"] for r in results if r.get("length", float('inf')) < float('inf')]
    lengths = [r["length"] for r in results if r.get("length", float('inf')) < float('inf')]
    times = [r.get("time", 0) for r in results if r.get("length", float('inf')) < float('inf')]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    # 左图：路径长度
    bars = ax1.bar(range(len(methods)), lengths, color=colors, edgecolor='white', linewidth=1)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('路径长度 (km)')
    ax1.set_title('解质量对比')

    if opt_length:
        ax1.axhline(y=opt_length, color='red', linestyle='--', linewidth=1,
                    label=f'最优解: {opt_length:.2f}')
        ax1.legend()

    for bar, val in zip(bars, lengths):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', fontsize=8)

    # 右图：运行时间（对数坐标）
    ax2.bar(range(len(times)), times, color=colors, edgecolor='white', linewidth=1)
    ax2.set_xticks(range(len(times)))
    ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('运行时间 (秒)')
    ax2.set_yscale('log')
    ax2.set_title('运行时间对比 (对数坐标)')

    for i, t in enumerate(times):
        ax2.text(i, t * 1.5, f'{t:.4f}s', ha='center', fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scalability(
    sizes: List[int],
    all_results: Dict[str, List[float]],
    metric: str = "length",
    title: str = "算法可扩展性分析",
    save_path: Optional[str] = None,
):
    """
    折线图展示算法随规模变化的性能

    参数:
        sizes: 配送点数量列表
        all_results: {算法名: [指标值列表]}
        metric: "length" 或 "time"
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for (method, values), color, marker in zip(all_results.items(), colors, markers):
        ax.plot(sizes, values, marker=marker, color=color,
                linewidth=2, markersize=8, label=method)

    ax.set_xlabel('配送点数量', fontsize=12)
    if metric == "length":
        ax.set_ylabel('路径长度 (km)', fontsize=12)
    else:
        ax.set_ylabel('运行时间 (秒)', fontsize=12)
        ax.set_yscale('log')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence(
    history: List[float],
    title: str = "模拟退火收敛曲线",
    save_path: Optional[str] = None,
):
    """绘制模拟退火收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history, 'b-', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('路径长度 (km)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注最小值
    min_idx = np.argmin(history)
    min_val = history[min_idx]
    ax.annotate(f'最优: {min_val:.2f}',
                xy=(min_idx, min_val),
                xytext=(min_idx + len(history) * 0.1, min_val + 5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    from data_generator import generate_random_scenario
    from tsp_algorithms import TSPSolver, compare_algorithms

    scenario = generate_random_scenario(15, seed=42)
    solver = TSPSolver(scenario.distance_matrix)

    # 测试单条路线绘制
    result = solver.solve("christofides+2opt")
    plot_route(scenario, result["tour"], "Christofides + 2-Opt 路线图",
               save_path="test_route.png")

    print("Test visualization saved to test_route.png")
