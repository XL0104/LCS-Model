
import json
import sys
import math
import os
from pprint import pprint

import torch_geometric.utils
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import expm_multiply
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges, to_networkx, subgraph
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix
from retrofitting.train import retrofit
from torch.nn import Embedding


def hk_chopper_subgraph(src, dst, A, L, G, t=1.0, max_iter=5, conductance_threshold=0.3, node_features=None, y=1):
    """
    通过完全依赖热核扩散与Chopper算法提取高质量局部封闭子图
    """
    num_nodes = A.shape[0]
    
    # 1. 初始化热核向量并模拟扩散
    heat_vector = np.zeros(num_nodes)
    heat_vector[src] = 1.0
    heat_vector[dst] = 1.0
    
    # expm_multiply针对大型稀疏矩阵具有显著性能优势
    heat_scores = expm_multiply(-t * L, heat_vector)
    
    # 2. 获取高分扩散节点（预处理阈值）
    threshold = np.percentile(heat_scores, 80)
    candidate_nodes = np.where(heat_scores >= threshold)[0]
    
    # 强制将目标边端点纳入子图
    if src not in candidate_nodes:
        candidate_nodes = np.append(candidate_nodes, src)
    if dst not in candidate_nodes:
        candidate_nodes = np.append(candidate_nodes, dst)
        
    S = set(candidate_nodes)
    
    # 3. Chopper剪枝优化子图电导
    degrees = dict(G.degree())
    vol_G = sum(degrees.values())
    
    vol_S = sum(degrees[u] for u in S)
    boundary_edges = sum(1 for u in S for v in G.neighbors(u) if v not in S)
    
    def calc_conductance(v_s, b_e):
        v_c = vol_G - v_s
        if v_s == 0 or v_c == 0:
            return 1.0
        return b_e / min(v_s, v_c)

    for iteration in range(max_iter):
        conductance = calc_conductance(vol_S, boundary_edges)
        if conductance <= conductance_threshold or len(S) <= 2:
            break
            
        best_node = None
        best_conductance = float('inf')
        best_boundary_edges = 0
        best_vol_S = 0
        
        for u in S:
            if u == src or u == dst: # 端点不可被剪枝
                continue
                
            deg_u = degrees[u]
            deg_S_u = sum(1 for v in G.neighbors(u) if v in S)
            
            # 使用增量法则快速计算移除该节点后的图状态
            new_vol_S = vol_S - deg_u
            new_boundary_edges = boundary_edges + 2 * deg_S_u - deg_u
            
            new_cond = calc_conductance(new_vol_S, new_boundary_edges)
            
            if new_cond < best_conductance:
                best_conductance = new_cond
                best_node = u
                best_boundary_edges = new_boundary_edges
                best_vol_S = new_vol_S
                
        # 不断缩减直到电导率达到最优下界
        if best_conductance < conductance and best_node is not None:
            S.remove(best_node)
            vol_S = best_vol_S
            boundary_edges = best_boundary_edges
        else:
            break
            
    nodes = list(S)
    
    # 4. 动态重排Node List (为了适配下游DRNL要求src为0位，dst为1位)
    nodes.remove(src)
    nodes.remove(dst)
    nodes = [src, dst] + nodes
    
    # 5. 生成封闭子图 (移除目标边防止标签泄漏)
    subgraph = A[nodes, :][:, nodes]
    subgraph = subgraph.tolil()
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    subgraph = subgraph.tocsr()
    
    if node_features is not None:
        node_features = node_features[nodes]
        
    dists = [0] * len(nodes) # 采用占位符跳过hop标签，适配基于距离标记
    
    return nodes, subgraph, dists, node_features, y


def py_g_drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    # adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py
    # Double-radius node labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                             indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                             indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='trunc'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='trunc'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                node_id=node_ids, num_nodes=num_nodes)
    return data


def calc_ratio_helper(*args, **kwargs):
    print("calc_ratio 统计被禁用，因为全域已经替换为基于 HeatKernel 和 Chopper 的提取。")
    sys.exit(0)


def extract_enclosing_subgraphs(link_index, A, L, G_nx, x, y, node_label='drnl'):
    # Extract enclosing subgraphs from A for all links in link_index using HeatKernel Chopper.
    data_list = []

    for src, dst in tqdm(link_index.t().tolist()):
        tmp = hk_chopper_subgraph(src, dst, A, L, G_nx, node_features=x, y=y)
        data = construct_pyg_graph(*tmp, node_label)
        draw = False
        if draw:
            draw_graph(to_networkx(data))
        data_list.append(data)

    return data_list


def do_seal_edge_split(data):
    # this is for datasets involving the WalkPooling paper
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos.t()
    split_edge['train']['edge_neg'] = data.train_neg.t()
    split_edge['valid']['edge'] = data.val_pos.t()
    split_edge['valid']['edge_neg'] = data.val_neg.t()
    split_edge['test']['edge'] = data.test_pos.t()
    split_edge['test']['edge_neg'] = data.test_neg.t()
    return split_edge


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1, neg_ratio=1, data_passed=False):
    if not data_passed:
        data = dataset[0]
    else:
        # for flow involving SEAL datasets, we pass data in dataset arg directly
        data = dataset

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1) * neg_ratio)
    else:
        raise NotImplementedError('Fast split is untested and unsupported.')

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100, neg_ratio=1):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1) * neg_ratio)
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        # TODO: find out what dataset split prompts this flow
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def HeatKernel_Chopper(A, edge_index, t=1.0, max_iter=5, conductance_threshold=0.3):
    """
    热核扩散 + Chopper剪枝算法 (Heuristic 分数打分逻辑)
    """
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    
    scores = []
    j = 0
    
    # 构建拉普拉斯矩阵 L = D - A
    D_diag = np.array(A.sum(axis=1)).flatten()
    D = ssp.diags(D_diag)
    L = D - A
    
    # 只构建一次网络图，极大提升循环性能
    try:
        G = nx.from_scipy_sparse_matrix(A)
    except AttributeError:
        G = nx.from_scipy_sparse_array(A)
        
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        
        src = edge_index[0, i].item()
        
        # 热核扩散计算：初始态向量
        heat_vector = np.zeros(num_nodes)
        heat_vector[src] = 1.0
        
        # 使用 expm_multiply 能够仅计算向量随网络结构的演化，拒绝了稠密矩阵相乘导致的显存溢出！
        heat_scores = expm_multiply(-t * L, heat_vector)
        
        # Chopper剪枝：电导剪枝 + 迭代精炼
        heat_scores = chopper_pruning(G, heat_scores, src, max_iter, conductance_threshold)
        
        j = i
        while j < edge_index.shape[1] and edge_index[0, j] == src:
            j += 1
            
        all_dst = edge_index[1, i:j].cpu().numpy()
        cur_scores = heat_scores[all_dst]
        
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def chopper_pruning(G, heat_scores, src, max_iter=5, conductance_threshold=0.3):
    """
    Chopper剪枝算法：电导剪枝 + 迭代精炼（加速版）
    通过利用差量缓存方法对原版冗余遍历进行了 O(1) 优化
    """
    # 初始子图：基于热核分数阈值
    threshold = np.percentile(heat_scores, 80)
    candidate_nodes = np.where(heat_scores >= threshold)[0]
    
    # 确保源节点在候选集中
    if src not in candidate_nodes:
        candidate_nodes = np.append(candidate_nodes, src)
        
    # 构建初始子图
    S = set(candidate_nodes)
    
    # 预计算整个图的数据，缓存计算以避免嵌套遍历
    degrees = dict(G.degree())
    vol_G = sum(degrees.values())
    
    # 初始参数的预计算
    vol_S = sum(degrees[u] for u in S)
    boundary_edges = sum(1 for u in S for v in G.neighbors(u) if v not in S)
    
    def calc_conductance(v_s, b_e):
        v_c = vol_G - v_s
        if v_s == 0 or v_c == 0:
            return 1.0
        return b_e / min(v_s, v_c)

    for iteration in range(max_iter):
        # 计算当前子图的电导
        conductance = calc_conductance(vol_S, boundary_edges)
        
        if conductance <= conductance_threshold or len(S) <= 2:
            # 电导满足要求或子图过小，停止迭代
            break
            
        # 电导剪枝：移除导致高电导的边界节点
        best_node = None
        best_conductance = float('inf')
        best_boundary_edges = 0
        best_vol_S = 0
        
        # 寻找摘除后改善效果最佳的节点
        for u in S:
            deg_u = degrees[u]
            deg_S_u = sum(1 for v in G.neighbors(u) if v in S)
            
            # 使用 O(1) 状态差量递推法获取摘除新状态
            new_vol_S = vol_S - deg_u
            new_boundary_edges = boundary_edges + 2 * deg_S_u - deg_u
            
            new_cond = calc_conductance(new_vol_S, new_boundary_edges)
            
            if new_cond < best_conductance:
                best_conductance = new_cond
                best_node = u
                best_boundary_edges = new_boundary_edges
                best_vol_S = new_vol_S
                
        # 只要能使电导变小就持续优化
        if best_conductance < conductance:
            S.remove(best_node)
            vol_S = best_vol_S
            boundary_edges = best_boundary_edges
        else:
            break
            
    # 基于最终子图重新计算热核分数
    refined_scores = np.zeros_like(heat_scores)
    refined_scores[list(S)] = heat_scores[list(S)]
    
    return refined_scores


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def add_info(self, epochs, runs):
        self.epochs = epochs
        self.runs = runs

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'Highest Test: {result[argmax, 1]:.2f}', file=f)
            print(f'Average Test: {result.T[1].mean():.2f} ± {result.T[1].std():.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}', file=f)
            print(f'\n(Precision of 5)Highest Test: {r.mean():.5f} ± {r.std():.5f}\n', file=f)
            if hasattr(self, 'epochs'):
                # logger won't have epochs while running heuristic models
                r_revised = torch.reshape(result, (self.epochs * self.runs, 2))[:, 1]
                print(f'Average Test: {r_revised.mean():.2f} ± {r_revised.std():.2f}', file=f)


def draw_graph(graph):
    # helps draw a graph object and save it as a png file
    f = plt.figure(1, figsize=(48, 48))
    nx.draw(graph, with_labels=True, pos=nx.spring_layout(graph))
    plt.show()  # check if same as in the doc visually
    f.savefig("input_graph.pdf", bbox_inches='tight')


# https://stackoverflow.com/a/45846841/12918863
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def n2v_emb(args, data, device):
    embedding = torch.load('embedding/embedding_custom.pt', map_location='cpu')
    if args.use_feature:
        model = retrofit(data, embedding, 0.1, 0.4, 0.1, 100)
        embedding = model(embedding)
    embedding = embedding.to(device)
    return embedding