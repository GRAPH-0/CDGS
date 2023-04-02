import os
import torch
import numpy as np
import random
import logging
import time
from absl import flags
from torch.utils import tensorboard
from torch_geometric.loader import DataLoader, DenseDataLoader
import pickle
from rdkit import RDLogger, Chem

from models import cdgs
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from evaluation import get_FCDMetric, get_nspdk_eval
import sde_lib
import visualize
from utils import *
from moses.metrics.metrics import get_all_metrics

FLAGS = flags.FLAGS


def set_random_seed(config):
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mol_sde_train(config, workdir):
    """Runs the training pipeline of molecule generation.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries.
            If this contains checkpoint training will be resumed from the latest checkpoint.
    """

    ### Ignore info output by RDKit
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build dataloader and iterators
    train_ds, eval_ds, test_ds, n_node_pmf = datasets.get_dataset(config)

    train_loader = DenseDataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    eval_loader = DenseDataLoader(eval_ds, batch_size=config.training.eval_batch_size, shuffle=False)
    test_loader = DenseDataLoader(test_ds, batch_size=config.training.eval_batch_size, shuffle=False)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(config.device)

    train_iter = iter(train_loader)
    # create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        atom_sde = sde_lib.VPSDE(beta_min=config.model.node_beta_min, beta_max=config.model.node_beta_max,
                                 N=config.model.num_scales)
        bond_sde = sde_lib.VPSDE(beta_min=config.model.edge_beta_min, beta_max=config.model.edge_beta_max,
                                 N=config.model.num_scales)
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn((atom_sde, bond_sde), train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn((atom_sde, bond_sde), train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    test_FCDMetric = get_FCDMetric(test_ds.sub_smiles, device=config.device)
    eval_FCDMetric = get_FCDMetric(eval_ds.sub_smiles, device=config.device)

    # Build sampling functions
    if config.training.snapshot_sampling:
        sampling_atom_shape = (config.training.eval_batch_size, config.data.max_node, config.data.atom_channels)
        sampling_bond_shape = (config.training.eval_batch_size, config.data.bond_channels,
                               config.data.max_node, config.data.max_node)
        sampling_fn = sampling.get_mol_sampling_fn(config, atom_sde, bond_sde, sampling_atom_shape, sampling_bond_shape,
                                                   inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        try:
            graphs = next(train_iter)
        except StopIteration:
            train_iter = train_loader.__iter__()
            graphs = next(train_iter)

        batch = dense_mol(graphs, scaler, config.data.dequantization)

        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            for eval_graphs in eval_loader:
                eval_batch = dense_mol(eval_graphs, scaler)
                eval_loss = eval_step_fn(state, eval_batch)
                logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
                writer.add_scalar("eval_loss", eval_loss.item(), step)
                break
            for test_graphs in test_loader:
                test_batch = dense_mol(test_graphs, scaler)
                test_loss = eval_step_fn(state, test_batch)
                logging.info("step: %d, test_loss: %.5e" % (step, test_loss.item()))
                writer.add_scalar("test_loss", test_loss.item(), step)
                break

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())

                atom_sample, bond_sample, sample_steps, sample_nodes = sampling_fn(score_model, n_node_pmf)

                sample_list, valid_wd = tensor2mol(atom_sample, bond_sample, sample_nodes, config.data.atom_list,
                                                   correct_validity=True, largest_connected_comp=True)
                ## fcd value
                smile_list = [Chem.MolToSmiles(mol) for mol in sample_list]
                fcd_test = test_FCDMetric(smile_list)
                fcd_eval = eval_FCDMetric(smile_list)

                ## log info
                valid_wd_rate = np.sum(valid_wd) / len(valid_wd)
                logging.info("step: %d, n_mol: %d, validity rate wd check: %.4f, fcd_val: %.4f, fcd_test: %.4f" %
                             (step, len(sample_list), valid_wd_rate, fcd_eval, fcd_test))

                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)
                # graph visualization and save figs
                visualize.visualize_mols(sample_list[:16], this_sample_dir, config)


def mol_sde_evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to "eval".
    """

    ### Ignore info output by RDKit
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build data pipeline
    train_ds, _, test_ds, n_node_pmf = datasets.get_dataset(config)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(config.device)
    # test_FCDMetric = get_FCDMetric(test_ds.sub_smiles, device=config.device)

    # Creat data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        atom_sde = sde_lib.VPSDE(beta_min=config.model.node_beta_min, beta_max=config.model.node_beta_max,
                                 N=config.model.num_scales)
        bond_sde = sde_lib.VPSDE(beta_min=config.model.edge_beta_min, beta_max=config.model.edge_beta_max,
                                 N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        atom_sde = sde_lib.subVPSDE(beta_min=config.model.node_beta_min, beta_max=config.model.node_beta_max,
                                    N=config.model.num_scales)
        bond_sde = sde_lib.subVPSDE(beta_min=config.model.edge_beta_min, beta_max=config.model.edge_beta_nax,
                                    N=config.model.num_scales)
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")


    if config.eval.enable_sampling:
        sampling_atom_shape = (config.eval.batch_size, config.data.max_node, config.data.atom_channels)
        sampling_bond_shape = (config.eval.batch_size, config.data.bond_channels,
                               config.data.max_node, config.data.max_node)
        sampling_fn = sampling.get_mol_sampling_fn(config, atom_sde, bond_sde, sampling_atom_shape, sampling_bond_shape,
                                                   inverse_scaler, sampling_eps)

    # Begin evaluation
    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))

    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(score_model.parameters())

        # Generate samples and compute MMD stats
        if config.eval.enable_sampling:
            num_sampling_rounds = int(np.ceil(config.eval.num_samples / config.eval.batch_size))
            all_samples = []
            all_valid_wd = []
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
                atom_sample, bond_sample, sample_steps, sample_nodes = sampling_fn(score_model, n_node_pmf)
                logging.info("sample steps: %d" % sample_steps)

                sample_list, valid_wd = tensor2mol(atom_sample, bond_sample, sample_nodes, config.data.atom_list,
                                                   correct_validity=True, largest_connected_comp=True)

                all_samples += sample_list
                all_valid_wd += valid_wd

            all_samples = all_samples[:config.eval.num_samples]
            all_valid_wd = all_valid_wd[:config.eval.num_samples]
            smile_list = []
            for mol in all_samples:
                if mol is not None:
                    smile_list.append(Chem.MolToSmiles(mol))

            # save the graphs
            sampler_name = config.sampling.method

            if config.eval.save_graph:
                # save the smile strings instead of rdkit mol object
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}.pkl".format(ckpt))
                with open(graph_file, "wb") as f:
                    pickle.dump(smile_list, f)

            # evaluate
            logging.info('Number of molecules: %d' % len(all_samples))
            ## valid, novelty, unique rate
            logging.info('sampling -- ckpt: {}, validity w/o correction: {:.6f}'.
                         format(ckpt, np.sum(all_valid_wd) / len(all_valid_wd)))

            ## moses metric
            scores = get_all_metrics(gen=smile_list, k=len(smile_list), device=config.device, n_jobs=8,
                                     test=test_ds.sub_smiles, train=train_ds.sub_smiles)
            for metric in ['valid', f'unique@{len(smile_list)}', 'FCD/Test', 'Novelty']:
                logging.info(f'sampling -- ckpt: {ckpt}, {metric}: {scores[metric]}')

            ## NSPDK evaluation
            if config.eval.nspdk:
                nspdk_eval = get_nspdk_eval(config)
                test_smiles = test_ds.sub_smiles
                test_mols = []
                for smile in test_smiles:
                    mol = Chem.MolFromSmiles(smile)
                    # Chem.Kekulize(mol)
                    test_mols.append(mol)
                test_nx_graphs = mols_to_nx(test_mols)
                gen_nx_graphs = mols_to_nx(all_samples)
                nspdk_res = nspdk_eval(test_nx_graphs, gen_nx_graphs)
                logging.info('sampling -- ckpt: {}, NSPDK: {}'.format(ckpt, nspdk_res))


run_train_dict = {
    'mol_sde': mol_sde_train
}
run_eval_dict = {
    'mol_sde': mol_sde_evaluate,
}


def train(config, workdir):
    run_train_dict[config.model_type](config, workdir)


def evaluate(config, workdir, eval_folder='eval'):
    run_eval_dict[config.model_type](config, workdir, eval_folder)
