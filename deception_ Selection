import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 定数の定義
NUM_NODES = 10
DECEPTION_BUDGET = 3
PROTECTION_BUDGET = 2

# グラフの生成
def generate_attack_graph(num_nodes):
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, reward=np.random.randint(1, 10))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() > 0.5:
                G.add_edge(i, j, success_prob=np.random.rand())
    return G

# 欺瞞行動の適用
def apply_deception(G, deception_budget):
    deception_actions = []
    edges = list(G.edges(data=True))
    while deception_budget > 0 and edges:
        edge = edges.pop(np.random.randint(0, len(edges)))
        if np.random.rand() > 0.5 and deception_budget >= 1:  # 隠蔽
            G.remove_edge(edge[0], edge[1])
            deception_actions.append(('hide', edge))
            deception_budget -= 1
        elif deception_budget >= 2:  # 偽エッジの追加
            new_edge = (edge[1], edge[0])
            G.add_edge(*new_edge, success_prob=0)
            deception_actions.append(('add', new_edge))
            deception_budget -= 2
    return deception_actions

# 防護戦略の計算
def calculate_protection(G, protection_budget):
    protection_actions = []
    edges = list(G.edges(data=True))
    for edge in edges:
        if protection_budget <= 0:
            break
        protect_amount = min(np.random.rand(), protection_budget)
        G[edge[0]][edge[1]]['protection'] = protect_amount
        protection_actions.append((edge, protect_amount))
        protection_budget -= protect_amount
    return protection_actions

# 攻撃者の最適反応のシミュレーション
def simulate_attacker(G):
    paths = list(nx.all_simple_paths(G, source=0, target=NUM_NODES-1))
    best_path = None
    best_reward = -np.inf
    for path in paths:
        reward = 0
        success_prob = 1.0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            success_prob *= G[edge[0]][edge[1]].get('success_prob', 0) * (1 - G[edge[0]][edge[1]].get('protection', 0))
            reward += G.nodes[path[i]]['reward']
        reward += G.nodes[path[-1]]['reward']
        if success_prob * reward > best_reward:
            best_reward = success_prob * reward
            best_path = path
    return best_path, best_reward

# グラフの可視化（同じレイアウトを使用）
def plot_graph_with_fixed_pos(G, pos, best_path=None, title="Attack Graph"):
    plt.figure(figsize=(10, 8))
    
    # ノードの描画
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # エッジの描画
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20)
    
    # ラベルの描画
    labels = {node: f"{node}\n{data['reward']}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    edge_labels = {(edge[0], edge[1]): f"{edge[2]['success_prob']:.2f}\n{edge[2].get('protection', 0):.2f}" for edge in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # 最適パスの描画
    if best_path:
        path_edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', arrowstyle='->', arrowsize=20, width=2)
    
    plt.title(title)
    plt.show()

# メインシミュレーション関数（固定レイアウトを使用）
def run_simulation_with_fixed_layout():
    G = generate_attack_graph(NUM_NODES)
    
    # 初期状態のグラフと攻撃者の最適パス
    best_path_initial, best_reward_initial = simulate_attacker(G)
    pos = nx.kamada_kawai_layout(G)  # レイアウトを固定
    plot_graph_with_fixed_pos(G, pos, best_path_initial, title="Initial Attack Graph and Attacker's Best Path")
    
    # 欺瞞行動と防護戦略の適用
    deception_actions = apply_deception(G, DECEPTION_BUDGET)
    protection_actions = calculate_protection(G, PROTECTION_BUDGET)

    # 変更後のグラフと攻撃者の最適パス
    best_path_final, best_reward_final = simulate_attacker(G)
    plot_graph_with_fixed_pos(G, pos, best_path_final, title="Modified Attack Graph and Attacker's Best Path")

    return best_path_initial, best_reward_initial, best_path_final, best_reward_final, deception_actions, protection_actions

# シミュレーションの実行
initial_path_fixed, initial_reward_fixed, final_path_fixed, final_reward_fixed, deceptions_fixed, protections_fixed = run_simulation_with_fixed_layout()
initial_path_fixed, initial_reward_fixed, final_path_fixed, final_reward_fixed, deceptions_fixed, protections_fixed
