import torch
import os
import logging
import re
import copy
import numpy as np
import torch.nn.functional as F
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Descriptors import MolLogP, qed
from sascorer import calculateScore

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


@torch.no_grad()
def dense_mol(graph_data, scaler=None, dequantization=False):
    """Extract features and masks from PyG Dense DataBatch.

    Args:
        graph_data: DataBatch object.
            y: [B, 1] graph property values.
            num_atom: [B, 1] number of atoms in graphs.
            smile: [B] smile sequences.
            x: [B, max_node, channel1] atom type features.
            adj: [B, channel2, max_node, max_node] bond type features.
            atom_mask: [B, max_node]

    Returns:
        atom_feat: [B, max_node, channel1]
        atom_mask: [B, max_node]
        bond_feat: [B, channel2, max_node, max_node]
        bond_mask: [B, 1, max_node, max_node]
    """

    atom_feat = graph_data.x
    bond_feat = graph_data.adj
    atom_mask = graph_data.atom_mask
    if len(atom_mask.shape) == 1:
        atom_mask = atom_mask.unsqueeze(0)
    bond_mask = (atom_mask[:, None, :] * atom_mask[:, :, None]).unsqueeze(1)
    bond_mask = torch.tril(bond_mask, -1)
    bond_mask = bond_mask + bond_mask.transpose(-1, -2)

    if dequantization:
        atom_noise = torch.rand_like(atom_feat)
        atom_feat = (atom_feat + atom_noise) / 2. * atom_mask[:, :, None]
        bond_noise = torch.rand_like(bond_feat)
        bond_noise = torch.tril(bond_noise, -1)
        bond_noise = bond_noise + bond_noise.transpose(1, 2)
        bond_feat = (bond_feat + bond_noise) / 2. * bond_mask

    atom_feat = scaler(atom_feat, atom=True)
    bond_feat = scaler(bond_feat, atom=False)

    return atom_feat * atom_mask.unsqueeze(-1), atom_mask, bond_feat * bond_mask, bond_mask


def adj2graph(adj, sample_nodes):
    """Covert the PyTorch tensor adjacency matrices to numpy array.

    Args:
        adj: [Batch_size, channel, Max_node, Max_node], assume channel=1
        sample_nodes: [Batch_size]
    """
    adj_list = []
    # discretization
    adj[adj >= 0.5] = 1.
    adj[adj < 0.5] = 0.
    for i in range(adj.shape[0]):
        adj_tmp = adj[i, 0]
        # symmetric
        adj_tmp = torch.tril(adj_tmp, -1)
        adj_tmp = adj_tmp + adj_tmp.transpose(0, 1)
        # truncate
        adj_tmp = adj_tmp.cpu().numpy()[:sample_nodes[i], :sample_nodes[i]]
        adj_list.append(adj_tmp)

    return adj_list


def quantize_mol(adjs):
    # Quantize generated molecules [B, 1, N, N]
    adjs = adjs.squeeze(1)
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)
    adjs = adjs * 3
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return np.array(adjs.to(torch.int64))


def quantize_mol_2(adjs):
    # Quantize generated molecules [B, 2, N, N]
    # The 2nd channel: 0 -> edge type; 1 -> edge existence
    if type(adjs).__name__ == 'Tensor':
        adjs = adjs.detach().cpu()
    else:
        adjs = torch.tensor(adjs)

    adj_0 = adjs[:, 0, :, :]
    adj_1 = adjs[:, 1, :, :]

    adj_0 = adj_0 * 3
    adj_0[adj_0 >= 2.5] = 3
    adj_0[torch.bitwise_and(adj_0 >= 1.5, adj_0 < 2.5)] = 2
    adj_0[torch.bitwise_and(adj_0 >= 0.5, adj_0 < 1.5)] = 1
    adj_0[adj_0 < 0.5] = 0

    adj_1[adj_1 < 0.5] = 0
    adj_1[adj_1 >= 0.5] = 1

    adjs = adj_0 * adj_1
    return np.array(adjs.to(torch.int64))


def construct_mol(x, A, num_node, atomic_num_list):
    mol = Chem.RWMol()
    atoms = np.argmax(x, axis=1)
    atoms = atoms[:num_node]

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    if len(A.shape) == 2:
        adj = A[:num_node, :num_node]
    elif A.shape[0] == 4:
        # A (edge_type, max_num_node, max_num_node)
        adj = np.argmax(A, axis=0)
        adj = np.array(adj)
        adj = adj[:num_node, :num_node]

        # Note. 3 means no existing edge (when constructing adj matrices)
        adj[adj == 3] = -1
        adj += 1
        adj = adj - np.eye(num_node)
    else:
        raise ValueError('Wrong Adj shape.')

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            # remove formal charge for fair comparison with GraphAF, GraphDF, GraphCNF

            # add formal charge to atom: e.g. [O+], [N+], [S+], not support [O-], [N-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)

    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible valency

    Return:
        True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(mol):
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            queue = []

            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])

    return mol, no_correct


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_chemical_validity(mol):
    """
    Check the chemical validity of the mol object. Existing mol object is not modified.

    Args: mol: Rdkit mol object

    Return:
          True if chemically valid, False otherwise
    """

    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False


def tensor2mol(x_atom, x_bond, num_atoms, atomic_num_list, correct_validity=True, largest_connected_comp=True):
    """Construct molecules from the atom and bond tensors.

    Args:
        x_atom: The node tensor [`number of samples`, `maximum number of atoms`, `number of possible atom types`].
        x_bond: The adjacency tensor [`number of samples`, `number of possible bond type`, `maximum number of atoms`,
            `maximum number of atoms`]
        num_atoms: The number of nodes for every sample [`number of samples`]
        atomic_num_list: A list to specify what each atom channel corresponds to.
        correct_validity: Whether to use the validity correction introduced by `MoFlow`.
        largest_connected_comp: Whether to use the largest connected component as the final molecule in the validity
            correction.

    Return:
        The list of Rdkit mol object. The check_chemical_validity rate without check.
    """
    if x_bond.shape[1] == 1:
        x_bond = quantize_mol(x_bond)
    elif x_bond.shape[1] == 2:
        x_bond = quantize_mol_2(x_bond)
    else:
        x_bond = x_bond.cpu().detach().numpy()

    x_atom = x_atom.cpu().detach().numpy()
    num_nodes = num_atoms.cpu().detach().numpy()

    gen_mols = []
    valid_cum = []

    for atom_elem, bond_elem, num_node in zip(x_atom, x_bond, num_nodes):
        mol = construct_mol(atom_elem, bond_elem, num_node, atomic_num_list)

        if correct_validity:
            # correct the invalid molecule
            cmol, no_correct = correct_mol(mol)
            if no_correct:
                valid_cum.append(1)
            else:
                valid_cum.append(0)
            vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
            gen_mols.append(vcmol)
        else:
            gen_mols.append(mol)

    return gen_mols, valid_cum


def penalized_logp(mol):
    """
    Calculate the reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Args:
        mol: Rdkit mol object

    Returns:
        :class:`float`
    """

    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def get_mol_qed(mol):
    return qed(mol)


def calculate_min_plogp(mol):
    """
    Calculate the reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`float`
    """

    p1 = penalized_logp(mol)
    s1 = Chem.MolToSmiles(mol, isomericSmiles=True)
    s2 = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    p2 = penalized_logp(mol1)
    p3 = penalized_logp(mol2)
    final_p = min(p1, p2)
    final_p = min(final_p, p3)
    return final_p


def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048, useChirality=True):
    """
    Calculate the similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule.

    Args:
        mol: Rdkit mol object
        target: Rdkit mol object

    Returns:
        :class:`float`, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target, radius=radius, nBits=nBits,
                                                            useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)


def convert_radical_electrons_to_hydrogens(mol):
    """
    Convert radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Return a new mol object.

    Args:
        mol: Rdkit mol object

    :rtype:
        Rdkit mol object
    """

    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        print('converting radical electrons to H')
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def get_final_smiles(mol):
    """
    Returns a SMILES of the final molecule. Converts any radical
    electrons into hydrogens. Works only if molecule is valid
    :return: SMILES
    """
    m = convert_radical_electrons_to_hydrogens(mol)
    return Chem.MolToSmiles(m, isomericSmiles=True)


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
            #    atomic_num=atom.GetAtomicNum(),
            #    formal_charge=atom.GetFormalCharge(),
            #    chiral_tag=atom.GetChiralTag(),
            #    hybridization=atom.GetHybridization(),
            #    num_explicit_hs=atom.GetNumExplicitHs(),
            #    is_aromatic=atom.GetIsAromatic())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
            #    bond_type=bond.GetBondType())

        nx_graphs.append(G)
    return nx_graphs
