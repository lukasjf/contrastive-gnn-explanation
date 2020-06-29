import random
from enum import Enum
from pathlib import Path

import networkx as nx
import typer
from tqdm import tqdm as tq


class Dataset(str, Enum):
    CYCLIQ = 'CYCLIQ'
    CYCLIQ_MULTI = 'CYCLIQ-MULTI'
    TRISQ = 'TRISQ'


def random_tree(n):
    g = nx.generators.trees.random_tree(n)
    for i in range(n):
        g.nodes[i]['label'] = 0
    return g


def attach_cycle(g, cycle_len, label, is_clique):
    N = len(g.nodes())
    host_cands = [k for k, v in g.nodes(data=True) if v['label'] == 0]
    host_node = random.choice(host_cands)
    neighbors = list(g.neighbors(host_node))
    for u in neighbors:
        g.remove_edge(u, host_node)

    # add the cycle
    cycle_nodes = [host_node]
    for i in range(cycle_len - 1):
        g.add_edge(cycle_nodes[-1], N + i)
        cycle_nodes.append(N + i)
    g.add_edge(host_node, cycle_nodes[-1])

    if is_clique:
        for u in cycle_nodes:
            for v in cycle_nodes:
                if u != v:
                    g.add_edge(u, v)

    for u in cycle_nodes:
        g.nodes[u]['label'] = label

    # restore host_node edges
    for u in neighbors:
        v = random.choice(cycle_nodes)
        g.add_edge(u, v)
    return g


def attach_cycles(g, cycle_len, count, is_clique=False):
    for i in range(count):
        attach_cycle(g, cycle_len, '%d-%d-%d' % (cycle_len, is_clique, i), is_clique)
    return g


def add_to_list(graph_list, g, label):
    graph_num = len(graph_list) + 1
    for u in g.nodes():
        g.nodes()[u]['graph_num'] = graph_num
    g.graph['graph_num'] = graph_num
    graph_list.append((g, label))


def trisq():
    all_graphs = []
    random.seed(0)
    for i in range(1000):
        g = random_tree(random.randint(8, 15))
        add_to_list(all_graphs, g, 0)
    random.seed(1)
    for i in range(1000):
        g = random_tree(random.randint(8, 15))
        count = random.randint(1, 4)
        attach_cycles(g, cycle_len=3, count=count)
        add_to_list(all_graphs, g, 1)
    random.seed(2)
    for i in range(1000):
        g = random_tree(random.randint(8, 15))
        count = random.randint(1, 4)
        attach_cycles(g, cycle_len=4, count=count)
        add_to_list(all_graphs, g, 2)
    random.seed(3)
    for i in range(1000):
        g = random_tree(random.randint(8, 15))
        count_tri = random.randint(1, 4)
        count_sq = random.randint(1, 4)
        attach_cycles(g, cycle_len=3, count=count_tri)
        attach_cycles(g, cycle_len=4, count=count_sq)
        add_to_list(all_graphs, g, 3)
    return all_graphs


def cycliq(is_multi):
    all_graphs = []
    label = 0
    if is_multi:
        random.seed(0)
        for i in range(1000):
            g = random_tree(random.randint(8, 15))
            add_to_list(all_graphs, g, label)
        label += 1
    random.seed(1)
    for i in range(1000):
        g = random_tree(random.randint(8, 15))
        count = random.randint(1, 2)
        attach_cycles(g, cycle_len=5, count=count)
        add_to_list(all_graphs, g, label)
    label += 1
    random.seed(2)
    for i in range(1000):
        g = random_tree(random.randint(8, 15))
        count = random.randint(1, 2)
        attach_cycles(g, cycle_len=5, count=count, is_clique=True)
        add_to_list(all_graphs, g, label)
    label += 1
    if is_multi:
        random.seed(3)
        for i in range(1000):
            g = random_tree(random.randint(8, 15))
            count = random.randint(1, 2)
            attach_cycles(g, cycle_len=5, count=count, is_clique=True)
            count = random.randint(1, 2)
            attach_cycles(g, cycle_len=5, count=count)
            add_to_list(all_graphs, g, label)
    return all_graphs


def write_gexf(output_path: Path, dataset: Dataset, graphs):
    path = output_path / dataset.value
    path.mkdir(exist_ok=True)
    print('Created .gexf files in %s' % path)
    for g, label in graphs:
        nx.write_gexf(g, path / ('%d.%d.gexf' % (g.graph['graph_num'], label)))


def write_adjacency(output_path: Path, dataset: Dataset, graphs):
    relabled_gs = []
    first_label = 1
    graph_indicator = []
    for g, label in tq(graphs):
        relabled_gs.append(nx.convert_node_labels_to_integers(g, first_label=first_label))
        N = len(g.nodes())
        first_label += N
        graph_indicator.extend([g.graph['graph_num']] * N)
    with open(output_path / ('%s_A.txt' % dataset.value), 'w') as f:
        for g in relabled_gs:
            for u, v in g.edges():
                f.write(f'{u}, {v}\n{v}, {u}\n')
    with open(output_path / ('%s_graph_indicator.txt' % dataset.value), 'w') as f:
        f.write('\n'.join(map(str, graph_indicator)))
    with open(output_path / ('%s_graph_labels.txt' % dataset.value), 'w') as f:
        f.write('\n'.join([str(label) for g, label in graphs]))


def main(dataset: Dataset, output_path: Path):
    print('Generating %s dataset' % dataset.value)
    if dataset == Dataset.CYCLIQ:
        graphs = cycliq(is_multi=False)
    elif dataset == Dataset.CYCLIQ_MULTI:
        graphs = cycliq(is_multi=True)
    elif dataset == Dataset.TRISQ:
        graphs = trisq()

    if not output_path.exists():
        typer.confirm("Output path does not exist, do you want to create it?", abort=True)
        output_path.mkdir()

    write_gexf(output_path, dataset, graphs)
    write_adjacency(output_path, dataset, graphs)


if __name__ == '__main__':
    typer.run(main)
