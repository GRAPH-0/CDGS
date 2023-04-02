import ast
import torch
import json
import os
import numpy as np
import os.path as osp
import pandas as pd
import pickle as pk
from itertools import repeat
from rdkit import Chem
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx, degree, to_networkx


bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""

    centered = config.data.centered
    if hasattr(config.data, "shift"):
        shift = config.data.shift
    else:
        shift = 0.

    if hasattr(config.data, 'norm'):
        atom_norm, bond_norm = config.data.norm
        assert shift == 0.

        def scale_fn(x, atom=False):
            if centered:
                x = x * 2. - 1.
            else:
                x = x
            if atom:
                x = x * atom_norm
            else:
                x = x * bond_norm
            return x
        return scale_fn
    else:
        if centered:
            # Rescale to [-1, 1]
            return lambda x: x * 2. - 1. + shift
        else:
            assert shift == 0.
            return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    centered = config.data.centered
    if hasattr(config.data, "shift"):
        shift = config.data.shift
    else:
        shift = 0.

    if hasattr(config.data, 'norm'):
        atom_norm, bond_norm = config.data.norm

        assert shift == 0.

        def inverse_scale_fn(x, atom=False):
            if atom:
                x = x / atom_norm
            else:
                x = x / bond_norm
            if centered:
                x = (x + 1.) / 2.
            else:
                x = x
            return x

        return inverse_scale_fn
    else:
        if centered:
            # Rescale [-1, 1] to [0, 1]
            return lambda x: (x + 1. - shift) / 2.
        else:
            assert shift == 0.
            return lambda x: x


def networkx_graphs(dataset):
    return [to_networkx(dataset[i], to_undirected=True, remove_self_loops=True) for i in range(len(dataset))]


class StructureDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 dataset_name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.dataset_name = dataset_name

        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not os.path.exists(self.raw_paths[0]):
            raise ValueError("Without raw files.")
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_file_names(self):
        return [self.dataset_name + '.pkl']

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']

    @property
    def num_node_features(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.dataset_name}({arg_repr})'

    def process(self):
        # Read data into 'Data' list
        input_path = self.raw_paths[0]
        with open(input_path, 'rb') as f:
            graphs_nx = pk.load(f)
        data_list = [from_networkx(G) for G in graphs_nx]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    @torch.no_grad()
    def max_degree(self):
        data_list = [self.get(i) for i in range(len(self))]

        def graph_max_degree(g_data):
            return max(degree(g_data.edge_index[1], num_nodes=g_data.num_nodes))

        degree_list = [graph_max_degree(data) for data in data_list]
        return int(max(degree_list).item())

    def n_node_pmf(self):
        node_list = [self.get(i).num_nodes for i in range(len(self))]
        n_node_pmf = np.bincount(node_list)
        n_node_pmf = n_node_pmf / n_node_pmf.sum()
        return n_node_pmf


class MolDataset(InMemoryDataset):
    # from DIG: Dive into Graphs
    """
        A Pytorch Geometric data interface for datasets used in molecule generation.

        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`.
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same property
            label when the processed dataset is used. You can change the augment :obj:`processed_filename`
            to re-process the dataset with intended property.

        Args:
            root (string, optional): Root directory where the dataset should be saved.
            name (string, optional): The name of the dataset. Available dataset names are as follows:
                                    :obj:`zinc250k`, :obj:`zinc_800_graphaf`, :obj:`zinc_800_jt`,
                                    :obj:`zinc250k_property`, :obj:`qm9_property`, :obj:`qm9`, :obj:`moses`.
            bond_ch (int): The channels for bond matrices. {1, 2, 4}
            prop_name (string, optional): The molecular property desired and used as the optimization target.
                                        (eg. "obj:`penalized_logp`)
            conf_dict (dictionary, optional): dictionary that stores all the configuration for the corresponding dataset
            transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data`
                object and returns a transformed version. The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data`
                object and returns a transformed version.The data object will be transformed before being saved to disk.
            pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and
                returns a boolean value, indicating whether the data object should be included in the final dataset.

    """

    def __init__(self, root, name, bond_ch, prop_name='penalized_logp',
                 conf_dict=None, transform=None, pre_transform=None,
                 pre_filter=None, processed_filename='data.pt'):

        self.processed_filename = processed_filename
        self.root = root
        self.name = name
        self.prop_name = prop_name
        self.bond_ch = bond_ch

        if conf_dict is None:
            config_file = pd.read_csv(os.path.join(os.path.dirname(__file__), 'mol_config.csv'), index_col=0)
            if self.name not in config_file:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(config_file.keys())
                raise ValueError(error_mssg)
            config = config_file[self.name]
        else:
            config = conf_dict

        self.url = config['url']
        self.available_prop = str(prop_name) in ast.literal_eval(config['prop_list'])
        self.smile_col = config['smile']
        self.num_max_node = int(config['num_max_node'])
        self.atom_list = ast.literal_eval(config['atom_list'])

        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not osp.exists(self.raw_paths[0]):
            self.download()
        if osp.exists(self.processed_paths[0]):
            self.data, self.slices, self.all_smiles = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return self.processed_filename

    def download(self):
        print('making raw files:', self.raw_dir)
        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        url = self.url
        path = download_url(url, self.raw_dir)

    def process(self):
        """Process the dataset from raw data file to the :obj:`self.processed_dir` folder."""

        print('Processing...')
        self.data, self.slices = self.pre_process()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        torch.save((self.data, self.slices, self.all_smiles), self.processed_paths[0])
        print('Done!')

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        """Get the data object at index idx. """

        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        data['smile'] = self.all_smiles[idx]

        if self.bond_ch == 1:
            with torch.no_grad():
                adj = data.adj
                ch = adj.shape[0]
                adj = torch.argmax(adj, dim=0)
                adj[adj == 3] = -1
                adj = (adj + 1).float()
                data['adj'] = adj.unsqueeze(0) / (ch - 1)
        elif self.bond_ch == 2:
            with torch.no_grad():
                adj = data.adj
                ch = adj.shape[0]
                adj = torch.argmax(adj, dim=0)
                adj[adj == 3] = -1
                adj_1 = ((adj + 1) != 0).float()
                adj = (adj + 1).float()
                adj = torch.stack([adj / (ch - 1), adj_1])
                data['adj'] = adj

        return data

    def pre_process(self):
        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=',', dtype='str')
        smile_list = list(input_df[self.smile_col])
        if self.available_prop:
            prop_list = list(input_df[self.prop_name])

        self.all_smiles = smile_list
        data_list = []

        for i in range(len(smile_list)):

            smile = smile_list[i]
            mol = Chem.MolFromSmiles(smile)
            Chem.Kekulize(mol)
            num_atom = mol.GetNumAtoms()
            if num_atom > self.num_max_node:
                continue
            else:
                # atoms
                atom_array = np.zeros((self.num_max_node, len(self.atom_list)), dtype=np.float32)
                atom_mask = np.zeros(self.num_max_node, dtype=np.float32)
                atom_mask[:num_atom] = 1.

                atom_idx = 0
                for atom in mol.GetAtoms():
                    atom_feature = atom.GetAtomicNum()
                    atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1
                    atom_idx += 1

                x = torch.tensor(atom_array)

                # bonds
                adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()

                    ch = bond_type_to_int[bond_type]
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    adj_array[ch, i, j] = 1.
                    adj_array[ch, j, i] = 1.

                adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
                # adj_array += np.eye(self.num_max_node)

                data = Data(x=x)
                data.adj = torch.tensor(adj_array)
                data.num_atom = num_atom
                data.atom_mask = torch.tensor(atom_mask)
                if self.available_prop:
                    data.y = torch.tensor([float(prop_list[i])])
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices

    def get_split_idx(self):
        """
        Gets the train-valid set split indices of the dataset.
        Return:
            A dictionary for training-validation split with key `train_idx` and `valid_idx`.
        """

        if self.name.find('zinc250k') != -1:
            path = os.path.join(self.root, 'raw/valid_idx_zinc250k.json')
            with open(path) as f:
                valid_idx = json.load(f)

        elif self.name.find('qm9') != -1:
            path = os.path.join(self.root, 'raw/valid_idx_qm9.json')
            with open(path) as f:
                valid_idx = json.load(f)['valid_idxs']
                valid_idx = list(map(int, valid_idx))

        else:
            print('No available split file for this dataset, please check.')
            return None

        train_idx = list(set(np.arange(self.__len__())).difference(set(valid_idx)))

        return {'train_idx': torch.tensor(train_idx, dtype=torch.long),
                'valid_idx': torch.tensor(valid_idx, dtype=torch.long)}

    def n_node_pmf(self):
        # if 'qm9' in self.name:
        #     n_node_pmf = [0. for _ in range(10)]
        #     n_node_pmf[-1] = 1.
        #     return np.array(n_node_pmf)
        node_list = [self.get(i).num_atom.item() for i in range(len(self))]
        n_node_pmf = np.bincount(node_list)
        n_node_pmf = n_node_pmf / n_node_pmf.sum()
        return n_node_pmf


class QM9(MolDataset):
    def __init__(self, root='./', bond_ch=4, prop_name='penalized_logp', conf_dict=None, transform=None,
                 pre_transform=None, pre_filter=None, processed_filename='data.pt'):
        name = 'qm9_property'
        super(QM9, self).__init__(root, name, bond_ch, prop_name, conf_dict, transform, pre_transform, pre_filter,
                                  processed_filename)


class ZINC250k(MolDataset):
    """
    The attributes of the output data:
        x: the node features.
        y: the property labels for the graph.
        adj: the edge features in the form of dense adjacent matrices.
        batch: the assignment vector which maps each node to its respective graph identifier and can help reconstruct
            single graphs.
        num_atom: number of atoms for each graph.
        smile: original SMILE sequences for the graphs.
    """

    def __init__(self, root='./', bond_ch=4, prop_name='penalized_logp', conf_dict=None, transform=None,
                 pre_transform=None, pre_filter=None, processed_filename='data.pt'):
        name = 'zinc250k_property'
        super(ZINC250k, self).__init__(root, name, bond_ch, prop_name, conf_dict, transform, pre_transform, pre_filter,
                                       processed_filename)


class MOSES(MolDataset):
    def __init__(self, root='./', bond_ch=4, prop_name=None, conf_dict=None, transform=None, pre_transform=None, pre_filter=None,
                 processed_filename='data.pt'):
        name = 'moses'
        super(MOSES, self).__init__(root, name, bond_ch, prop_name, conf_dict, transform, pre_transform, pre_filter,
                                    processed_filename)


class ZINC800(MolDataset):
    """
    ZINC800 contains 800 selected molecules with lowest penalized logP scores. While method `jt` selects from the test
    set and `graphaf` selects from the train set.
    """

    def __init__(self, root='./', method='jt', bond_ch=4, prop_name='penalized_logp', conf_dict=None, transform=None,
                 pre_transform=None, pre_filter=None, processed_filename='data.pt'):
        name = 'zinc_800'
        name = name + '_' + method

        super(ZINC800, self).__init__(root, name, bond_ch, prop_name, conf_dict, transform, pre_transform, pre_filter,
                                      processed_filename)


def get_opt_dataset(config):
    """Create data loaders for similarity constrained molecule optimization.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.

    Returns:
        dataset
    """
    transform = T.Compose([
        T.ToDevice(config.device)
    ])

    assert 'zinc_800' in config.data.name

    if 'jt' in config.data.name:
        dataset = ZINC800(config.data.root, 'jt', bond_ch=config.data.bond_channels, transform=transform)
    elif 'graphaf' in config.data.name:
        dataset = ZINC800(config.data.root, 'graphaf', bond_ch=config.data.bond_channels, transform=transform)
    else:
        error_mssg = 'Invalid method type {}.\n'.format(config.data.name)
        error_mssg += 'Available datasets are as follows:\n'
        error_mssg += '\n'.join(['jt', 'graphaf'])
        raise ValueError(error_mssg)

    return dataset


def get_dataset(config):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.

    Returns:
        train_ds, eval_ds, test_ds, n_node_pmf
    """
    # define data transforms
    transform = T.Compose([
        # T.ToDense(config.data.max_node),
        T.ToDevice(config.device)
    ])

    # Build up data iterators
    if config.model_type == 'mol_sde' or config.model_type == 'sep_mol_sde':
        if config.data.name == 'QM9':
            dataset = QM9(config.data.root, bond_ch=config.data.bond_channels, transform=transform)
        elif config.data.name == 'ZINC250K':
            if hasattr(config.data, 'property'):
                property = config.data.property
            else:
                property = 'penalized_logp'
            if property == 'qed':
                dataset = ZINC250k(config.data.root, prop_name=property, bond_ch=config.data.bond_channels,
                                   transform=transform, processed_filename='qed_data.pt')
            else:
                dataset = ZINC250k(config.data.root, prop_name=property,
                                   bond_ch=config.data.bond_channels, transform=transform)
        elif config.data.name == 'MOSES':
            dataset = MOSES(config.data.root, bond_ch=config.data.bond_channels, transform=transform)
        else:
            raise ValueError('Undefined dataset name.')

        all_smiles = dataset.all_smiles
        splits = dataset.get_split_idx()
        train_idx = splits['train_idx']
        test_idx = splits['valid_idx']

        train_dataset = dataset[train_idx]
        train_dataset.sub_smiles = [all_smiles[idx] for idx in train_idx]
        test_dataset = dataset[test_idx]
        test_dataset.sub_smiles = [all_smiles[idx] for idx in test_idx]

        eval_idx = train_idx[torch.randperm(len(train_idx))[:len(test_idx)]]
        eval_dataset = dataset[eval_idx]
        eval_dataset.sub_smiles = [all_smiles[idx] for idx in eval_idx]
    else:
        dataset = StructureDataset(config.data.root, config.data.name, transform=transform)
        num_train = int(len(dataset) * config.data.split_ratio)
        num_test = len(dataset) - num_train
        train_dataset = dataset[:num_train]
        eval_dataset = dataset[:num_test]
        test_dataset = dataset[num_train:]

    n_node_pmf = train_dataset.n_node_pmf()

    return train_dataset, eval_dataset, test_dataset, n_node_pmf
