"""All functions related to loss computation and optimization."""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VPSDE


def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer


def optimization_manager(config):
    """Return an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimize with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_multi_sde_loss_fn(atom_sde, bond_sde, train, reduce_mean=True, continuous=True, eps=1e-5):
    """ Create a loss function for training with arbitrary node SDE and edge SDE.

        Args:
            atom_sde, bond_sde: An `sde_lib.SDE` object that represents the forward SDE.
            train: `True` for training loss and `False` for evaluation loss.
            reduce_mean: If `True`, average the loss across data dimensions. Otherwise, sum the loss across data dimensions.
            continuous: `True` indicates that the model is defined to take continuous time steps.
                        Otherwise, it requires ad-hoc interpolation to take continuous time steps.
            eps: A `float` number. The smallest time step to sample from.

        Returns:
            A loss function.
        """

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data, including node_features, adjacency matrices, node mask and adj mask.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """

        atom_feat, atom_mask, bond_feat, bond_mask = batch
        score_fn = mutils.get_multi_score_fn(atom_sde, bond_sde, model, train=train, continuous=continuous)
        t = torch.rand(atom_feat.shape[0], device=atom_feat.device) * (atom_sde.T - eps) + eps

        # perturbing atom
        z_atom = torch.randn_like(atom_feat)  # [B, N, C]
        mean_atom, std_atom = atom_sde.marginal_prob(atom_feat, t)
        perturbed_atom = (mean_atom + std_atom[:, None, None] * z_atom) * atom_mask[:, :, None]

        # perturbing bond
        z_bond = torch.randn_like(bond_feat)  # [B, C, N, N]
        z_bond = torch.tril(z_bond, -1)
        z_bond = z_bond + z_bond.transpose(-1, -2)
        mean_bond, std_bond = bond_sde.marginal_prob(bond_feat, t)
        perturbed_bond = (mean_bond + std_bond[:, None, None, None] * z_bond) * bond_mask

        atom_score, bond_score = score_fn((perturbed_atom, perturbed_bond), t, atom_mask=atom_mask, bond_mask=bond_mask)

        # atom loss
        atom_mask = atom_mask[:, :, None].repeat(1, 1, atom_feat.shape[-1])
        atom_mask = atom_mask.reshape(atom_mask.shape[0], -1)
        losses_atom = torch.square(atom_score * std_atom[:, None, None] + z_atom)
        losses_atom = losses_atom.reshape(losses_atom.shape[0], -1)
        if reduce_mean:
            losses_atom = torch.sum(losses_atom * atom_mask, dim=-1) / torch.sum(atom_mask, dim=-1)
        else:
            losses_atom = 0.5 * torch.sum(losses_atom * atom_mask, dim=-1)
        loss_atom = losses_atom.mean()

        # bond loss
        bond_mask = bond_mask.repeat(1, bond_feat.shape[1], 1, 1)
        bond_mask = bond_mask.reshape(bond_mask.shape[0], -1)
        losses_bond = torch.square(bond_score * std_bond[:, None, None, None] + z_bond)
        losses_bond = losses_bond.reshape(losses_bond.shape[0], -1)
        if reduce_mean:
            losses_bond = torch.sum(losses_bond * bond_mask, dim=-1) / (torch.sum(bond_mask, dim=-1) + 1e-8)
        else:
            losses_bond = 0.5 * torch.sum(losses_bond * bond_mask, dim=-1)
        loss_bond = losses_bond.mean()

        return loss_atom + loss_bond

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
             Tuple (`sde_lib.SDE`, `sde_lib.SDE`) that represents the forward node SDE and edge SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions.
            Otherwise, sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
            https://arxiv.org/abs/2101.09258; otherwise, use the weighting recommended by score-sde.

    Returns:
        A one-step function for training or evaluation.
    """

    if continuous:
        if isinstance(sde, tuple):
            loss_fn = get_multi_sde_loss_fn(sde[0], sde[1], train, reduce_mean=reduce_mean, continuous=True)
        else:
            loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                      continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, tuple):
            raise ValueError("Discrete training for multi sde is not recommended.")
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        For jax version: This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and
            jit-compiled together for faster execution.

        Args:
            state: A dictionary of training information, containing the score model, optimizer,
                EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data, including min-batch adjacency matrices and mask.

        Returns:
            loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn

