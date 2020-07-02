import os
import random
from enum import Enum
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import typer
from geomloss import SamplesLoss
from tqdm import tqdm as tq

from models import GcnEncoderGraph

app = typer.Typer(add_completion=False)


class ExplainMethod(str, Enum):
    contrastive = 'contrastive'
    sa = 'sensitivity'
    occlusion = 'occlusion'
    random = 'random'


def load_model(model_path: Path):
    ckpt = torch.load(model_path)
    cg_dict = ckpt["cg"]  # get computation graph
    input_dim = cg_dict["feat"].shape[2]
    num_classes = cg_dict["pred"].shape[2]
    model = GcnEncoderGraph(
        input_dim=input_dim,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        args=None,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def check_path(output_path: Path):
    if not output_path.exists():
        typer.confirm("Output path does not exist, do you want to create it?", abort=True)
        output_path.mkdir(parents=True)


def read_graphs(dataset_path: Path):
    labels = {}
    nx_graphs = {}
    for name in os.listdir(str(dataset_path)):
        if not name.endswith('gexf'):
            continue
        idx, label = name.split('.')[-3:-1]
        idx, label = int(idx), int(label)
        nx_graphs[idx] = nx.read_gexf(dataset_path / name)
        labels[idx] = label
    print('Found %d samples' % len(nx_graphs))
    return nx_graphs, labels


@app.command(name='sensitivity', help='Run sensitivity analysis explanation')
def sa(dataset_path: Path, model_path: Path, output_path: Path):
    check_path(output_path)
    nx_graphs, labels = read_graphs(dataset_path)
    model = load_model(model_path)

    def explain(graph_num):
        g = nx_graphs[graph_num]
        node_count = len(g.nodes)

        adj = np.zeros((1, 100, 100))
        adj[0, :node_count, :node_count] = nx.to_numpy_matrix(g)
        adj = torch.tensor(adj, dtype=torch.float)
        x = torch.ones((1, 100, 10), requires_grad=True, dtype=torch.float)

        ypred, _ = model(x, adj)

        loss = model.loss(ypred, torch.LongTensor([labels[graph_num]]))
        loss.backward()
        node_importance = x.grad.detach().numpy()[0][:node_count]
        node_importance = (node_importance ** 2).sum(axis=1)
        N = nx_graphs[graph_num].number_of_nodes()
        masked_adj = np.zeros((N, N))
        for u, v in nx_graphs[graph_num].edges():
            u = int(u)
            v = int(v)
            masked_adj[u, v] = masked_adj[v, u] = node_importance[u] + node_importance[v]
        return masked_adj

    for gid in tq(nx_graphs):
        masked_adj = explain(gid)
        np.save(output_path / ('%s.npy' % gid), masked_adj)


@app.command(help='Run occlusion explanation')
def occlusion(dataset_path: Path, model_path: Path, output_path: Path):
    check_path(output_path)
    nx_graphs, labels = read_graphs(dataset_path)
    model = load_model(model_path)

    def prepare_input(g):
        node_count = len(g.nodes)
        adj = np.zeros((1, 100, 100))
        adj[0, :node_count, :node_count] = nx.to_numpy_matrix(g)
        adj = torch.tensor(adj, dtype=torch.float)
        x = torch.ones((1, 100, 10), requires_grad=False, dtype=torch.float)
        return x, adj

    def explain(graph_num):
        model.eval()
        g = nx_graphs[graph_num]
        x, adj = prepare_input(g)

        ypred, _ = model(x, adj)
        true_label = labels[graph_num]
        before_occlusion = ypred[0].softmax(0)
        node_importance = {}

        for removed_node in g.nodes():
            g2 = g.copy()
            g2.remove_node(removed_node)
            x, adj = prepare_input(g2)
            ypred, _ = model(x, adj)
            after_occlusion = ypred[0].softmax(0)
            importance = abs(after_occlusion[true_label] - before_occlusion[true_label])
            node_importance[int(removed_node)] = importance.item()

        N = nx_graphs[graph_num].number_of_nodes()
        masked_adj = np.zeros((N, N))
        for u, v in nx_graphs[graph_num].edges():
            u = int(u)
            v = int(v)
            masked_adj[u, v] = masked_adj[v, u] = node_importance[u] + node_importance[v]
        return masked_adj

    for gid in tq(nx_graphs):
        masked_adj = explain(gid)
        np.save(output_path / ('%s.npy' % gid), masked_adj)


@app.command(name='random', help='Run random explanation')
def random_explain(dataset_path: Path, output_path: Path):
    check_path(output_path)
    nx_graphs, labels = read_graphs(dataset_path)

    def explain(graph_num):
        g = nx_graphs[graph_num]
        random_importance = list(range(len(g.edges())))
        random.shuffle(random_importance)

        N = g.number_of_nodes()
        masked_adj = np.zeros((N, N))
        for (u, v), importance in zip(g.edges(), random_importance):
            u = int(u)
            v = int(v)
            masked_adj[u, v] = masked_adj[v, u] = importance
        return masked_adj

    for gid in tq(nx_graphs):
        masked_adj = explain(gid)
        np.save(output_path / ('%s.npy' % gid), masked_adj)


@app.command(name='contrast', help='Run contrastive explanation')
def contrast(dataset_path: Path,
             embedding_path: Path = typer.Argument(..., help='path containing the graph embeddings'),
             output_path: Path = typer.Argument(..., help='output path for explanations'),
             loss_str: str = typer.Option('-+s', '--loss',
                                          help="add each of '-', '+' and 's' for different parts of loss. Order does not matter"),
             similar_size: int = typer.Option(10,
                                              help="number of similar graphs to use for positive and negative set"),
             distance_str: str = typer.Option('ot', '--distance',
                                              help="distance measure to use can be one of ['ot,'avg']")
             ):
    check_path(output_path)
    nx_graphs, labels = read_graphs(dataset_path)
    torch.set_num_threads(1)
    graph_embs = {}
    for name in os.listdir(str(embedding_path)):
        if not name.endswith('npy'):
            continue
        graph_num = int(name.split('.')[0])
        embs = np.load(str(embedding_path / name))
        last_idx = len(nx_graphs[graph_num].nodes)
        embs = embs[:last_idx, :]
        graph_embs[graph_num] = embs

    def closest(graph_num, dist, size=1, neg_label=None):
        cur_label = labels[graph_num]
        pos_dists = []
        neg_dists = []
        for i in graph_embs:
            if i == graph_num:
                continue
            #         if pred_labels[i] != dataset[i][1]: # ignore those not predicted correct
            #             continue
            d = dist(graph_num, i)
            if labels[i] != cur_label:
                if neg_label is None or labels[i] == neg_label:
                    neg_dists.append((d, i))
            else:
                pos_dists.append((d, i))
        pos_dists = sorted(pos_dists)
        neg_dists = sorted(neg_dists)
        pos_indices = [i for d, i in pos_dists]
        neg_indices = [i for d, i in neg_dists]

        return pos_indices[:size], neg_indices[:size]

    def loss_verbose(loss_str):
        res = ''
        if '-' in loss_str:
            res = res + '+ loss_neg '
        if '+' in loss_str:
            res = res + '- loss_pos '
        if 's' in loss_str:
            res = res + '+ loss_self '
        return res

    print('Using %s for loss function' % loss_verbose(loss_str))

    if distance_str == 'ot':
        distance = SamplesLoss("sinkhorn", p=1, blur=.01)
    elif distance_str == 'avg':
        distance = lambda x, y: torch.dist(x.mean(axis=0), y.mean(axis=0))

    def graph_distance(g1_num, g2_num):
        k = (min(g1_num, g2_num), max(g1_num, g2_num))
        g1_embs = graph_embs[g1_num]
        g2_embs = graph_embs[g2_num]
        return distance(torch.Tensor(g1_embs), torch.Tensor(g2_embs)).item()

    def explain(graph_num):
        cur_embs = torch.Tensor(graph_embs[graph_num])

        distance = SamplesLoss("sinkhorn", p=1, blur=.01)

        positive_ids, negative_ids = closest(graph_num, graph_distance, size=similar_size)

        positive_embs = [torch.Tensor(graph_embs[i]) for i in positive_ids]
        negative_embs = [torch.Tensor(graph_embs[i]) for i in negative_ids]

        mask = torch.nn.Parameter(torch.zeros(len(cur_embs)))

        learning_rate = 1e-1
        optimizer = torch.optim.Adam([mask], lr=learning_rate)

        if distance_str == 'ot':
            def mydist(mask, embs):
                return distance(mask.softmax(0), cur_embs,
                                distance.generate_weights(embs), embs)
        else:
            def mydist(mask, embs):
                return torch.dist((cur_embs * mask.softmax(0).reshape(-1, 1)).sum(axis=0), embs.mean(axis=0))
        # tq = tqdm(range(50))
        history = []
        for t in range(50):
            loss_pos = torch.mean(torch.stack([mydist(mask, x) for x in positive_embs]))
            loss_neg = torch.mean(torch.stack([mydist(mask, x) for x in negative_embs]))
            loss_self = mydist(mask, cur_embs)

            loss = 0
            if '-' in loss_str:
                loss = loss + loss_neg
            if '+' in loss_str:
                loss = loss - loss_pos
            if 's' in loss_str:
                loss = loss + loss_self

            hist_item = dict(loss_neg=loss_neg.item(), loss_self=loss_self.item(), loss_pos=loss_pos.item(),
                             loss=loss.item())
            history.append(hist_item)
            # tq.set_postfix(**hist_item)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        node_importance = list(1 - mask.softmax(0).detach().numpy().ravel())
        N = nx_graphs[graph_num].number_of_nodes()
        masked_adj = np.zeros((N, N))
        for u, v in nx_graphs[graph_num].edges():
            u = int(u)
            v = int(v)
            masked_adj[u, v] = masked_adj[v, u] = node_importance[u] + node_importance[v]
        return masked_adj

    for gid in tq(graph_embs):
        masked_adj = explain(gid)
        np.save(output_path / ('%s.npy' % gid), masked_adj)


if __name__ == "__main__":
    app()
