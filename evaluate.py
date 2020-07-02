import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer

from explain import read_graphs


def main(dataset_path: Path, explain_path: Path):
    nx_graphs, labels = read_graphs(dataset_path)

    graph_masked_adjs = {}

    name_map = {}
    for name in os.listdir(str(explain_path)):
        if not name.endswith('npy'):
            continue
        name_map[name.split('.')[0]] = name

    for i in nx_graphs:
        if str(i) not in name_map:
            continue
        masked_adj = np.load(str(explain_path / name_map[str(i)]))
        last_idx = len(nx_graphs[i].nodes)
        masked_adj = masked_adj[:last_idx, :last_idx]
        graph_masked_adjs[i] = masked_adj

    def explain(graph_num):
        edge_importance = {}
        for u, v in nx_graphs[graph_num].edges():
            u = int(u)
            v = int(v)
            edge_importance[(u, v)] = graph_masked_adjs[graph_num][u, v]
        return edge_importance

    def get_correct_edges(g):
        nodes_by_label = defaultdict(list)
        for u, data in g.nodes(data=True):
            nodes_by_label[data['label']].append(u)
        edges = []
        for label, nodes in nodes_by_label.items():
            if label == '0' or label == 0:
                continue
            edges.extend([(int(u), int(v)) for u, v in g.subgraph(nodes).edges()])
        return edges

    def get_accuracy(correct_edges, edge_importance):

        # Extract as many as correct edges
        predicted_edges = sorted(edge_importance.keys(), key=lambda e: -edge_importance[e])[:len(correct_edges)]
        correct = 0
        for u, v in predicted_edges:
            if (u, v) in correct_edges or (v, u) in correct_edges:
                correct += 1
        return correct / len(correct_edges)

    accs = []
    accs_by_label = defaultdict(list)
    for idx in nx_graphs:
        if str(idx) not in name_map:
            continue
        correct_edges = get_correct_edges(nx_graphs[idx])
        if len(correct_edges) == 0:
            continue
        edge_importance = explain(idx)
        acc = get_accuracy(correct_edges, edge_importance)
        accs.append(acc)
        accs_by_label[labels[idx]].append(acc)
    print('Total accuracy:')
    print('Sample count:', len(accs), 'Mean accuracy:', np.mean(accs), 'standard deviation:', np.std(accs))
    print('Accuracy by label:')
    for k, v in sorted(accs_by_label.items()):
        print('Accuracy for label', k)
        print('Sample count:', len(v), 'Mean accuracy:', np.mean(v), 'standard deviation:', np.std(v))
        print('-' * 40)


if __name__ == '__main__':
    app = typer.Typer(add_completion=False)
    app.command()(main)
    app()
