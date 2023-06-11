import os.path

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw
import matplotlib


def draw_graph_list(graph_list, row, col, f_path, iterations=100, layout='spring', is_single=False, k=1,
                    node_size=55, alpha=1, width=1.3, remove=True):

    G_list = [nx.to_networkx_graph(graph_list[i]) for i in range(len(graph_list))]

    # remove isolate nodes in graphs
    if remove:
        for gg in G_list:
            gg.remove_nodes_from(list(nx.isolates(gg)))

    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.axis("off")

        # turn off axis label
        plt.xticks([])
        plt.yticks([])

        if layout == 'spring':
            pos = nx.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=iterations)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(f'{layout} not recognized.')

        if is_single:
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0,
                                   )
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699', alpha=1, linewidths=0.2)
            # nx.draw_networkx_nodes(G, pos, node_size=2.0, node_color='#336699', alpha=1, linewidths=1.0)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)
            # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

    plt.tight_layout()
    plt.savefig(f_path, dpi=1600)
    plt.close()


def visualize_graphs(graph_list, dir_path, config, remove=True):
    row = config.sampling.vis_row
    col = config.sampling.vis_col
    n_graph = row * col

    n_fig = int(np.ceil(len(graph_list) / n_graph))
    for i in range(n_fig):
        draw_graph_list(graph_list[i*n_graph:(i+1)*n_graph], row, col,
                        f_path=os.path.join(dir_path, "sample"+str(i)+".png"), remove=remove)


def visualize_diff_graphs(graph_list, dir_path, sample_name, remove=True):
    draw_graph_list(graph_list, 4, 4, f_path=os.path.join(dir_path, sample_name+'.png'), remove=remove)


def draw_adjacency_matrix(adjs, times, fname='diff_test.png'):

    n_graphs = len(adjs)
    plt.figure(figsize=(10, 3), dpi=300)

    for i, adj in enumerate(adjs):
        adj = np.clip(adj, a_min=0., a_max=1.)
        plt.subplot(1, n_graphs, i+1)
        plt.axis('off')
        plt.title(str(times[i]))
        #plt.imshow(adj, cmap='binary', interpolation="none")
        #plt.imshow(adj, cmap='YlGn', interpolation="none")
        plt.imshow(adj, cmap='Reds', interpolation="none")

    # plt.savefig(fname, dpi=1600)
    plt.savefig(fname, dpi=300)


def draw_matrix(adjs, times, fname='diff_test.png'):

    n_graphs = len(adjs)
    plt.figure(figsize=(10, 3), dpi=300)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    for i, adj in enumerate(adjs):
        adj = np.clip(adj, a_min=0., a_max=1.)
        plt.subplot(1, n_graphs, i+1)
        plt.axis('off')
        plt.title(str(times[i]))
        #plt.imshow(adj, cmap='binary', interpolation="none")
        #plt.imshow(adj, cmap='YlGn', interpolation="none")
        plt.imshow(adj, cmap='Reds', interpolation="none", norm=norm)
        # plt.colorbar()

    # plt.savefig(fname, dpi=1600)
    plt.savefig(fname, dpi=300)


def visualize_mols(mol_list, dir_path, config):
    # from rdkit.Chem.Draw import IPythonConsole

    row = config.sampling.vis_row
    col = config.sampling.vis_col
    n_mol = row * col

    img = Draw.MolsToGridImage(mol_list[:n_mol], subImgSize=(400, 400), molsPerRow=row)
    img.save(os.path.join(dir_path, 'mol.png'))

