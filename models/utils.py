"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_mol_model(config):
    """Create the score model."""
    atom_name = config.model.atom_name
    atom_model = get_model(atom_name)(config)
    atom_model = atom_model.to(config.device)
    atom_model = torch.nn.DataParallel(atom_model)

    bond_name = config.model.bond_name
    bond_model = get_model(bond_name)(config)
    bond_model = bond_model.to(config.device)
    bond_model = torch.nn.DataParallel(bond_model)

    return atom_model, bond_model


def get_multi_score_fn(atom_sde, bond_sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

        Args:
            atom_sde: An `sde_lib.SDE` object that represents the forward SDE.
            bond_sde: An `sde_lib.SDE` object that represents the forward SDE.
            model: A score model.
            train: `True` for training and `False` for evaluation.
            continuous: If `True`, the score-based model is expected to directly take continuous time steps.

        Returns:
            A score function.
        """
    model_fn = get_model_fn(model, train=train)

    if isinstance(atom_sde, sde_lib.VPSDE) or isinstance(atom_sde, sde_lib.subVPSDE):
        def score_fn(x, t, *args, **kwargs):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for continuously-trained models.
                labels = t * 999
                atom_score, bond_score = model_fn(x, labels, *args, **kwargs)
                atom_std = atom_sde.marginal_prob(torch.zeros_like(x[0]), t)[1]
                bond_std = bond_sde.marginal_prob(torch.zeros_like(x[1]), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                atom_score, bond_score = model_fn(x, labels, *args, **kwargs)
                atom_std = atom_sde.sqrt_1m_alpha_cumprod.to(labels.device)[labels.long()]
                bond_std = bond_sde.sqrt_1m_alpha_cumprod.to(labels.device)[labels.long()]

            atom_score = -atom_score / atom_std[:, None, None]
            bond_score = -bond_score / bond_std[:, None, None, None]
            return atom_score, bond_score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=False):
    """Wraps `theta_fn` so that the model output corresponds to a real time-dependent score function.

        Args:
            atom_sde: An `sde_lib.SDE` object that represents the forward SDE.
            bond_sde: An `sde_lib.SDE` object that represents the forward SDE.
            model: A score model.
            train: `True` for training and `False` for evaluation.
            continuous: If `True`, the score-based model is expected to directly take continuous time steps.

        Returns:
            A theta function.
        """
    model_fn = get_model_fn(model, train=train)

    if isinstance(atom_sde, sde_lib.VPSDE) or isinstance(atom_sde, sde_lib.subVPSDE):
        def theta_fn(x, t, *args, **kwargs):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for continuously-trained models.
                labels = t * 999
                atom_theta, bond_theta = model_fn(x, labels, *args, **kwargs)
            else:
                raise NotImplementedError()

            return atom_theta, bond_theta

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return theta_fn


def get_mol_regressor_grad_fn(atom_sde, bond_sde, regressor_fn, norm=False):
    """Get the noise graph regressor gradient fn."""
    N = atom_sde.N - 1

    def mol_regressor_grad_fn(x, t, only_grad=False, std=False, *args, **kwargs):
        label = t * N
        atom_std = atom_sde.marginal_prob(torch.zeros_like(x[0]), t)[1]
        bond_std = bond_sde.marginal_prob(torch.zeros_like(x[1]), t)[1]

        with torch.enable_grad():
            atom_in, bond_in = x
            atom_in = atom_in.detach().requires_grad_(True)
            bond_in = bond_in.detach().requires_grad_(True)
            pred = regressor_fn((atom_in, bond_in), label, *args, **kwargs)
            try:
                atom_grad, bond_grad = torch.autograd.grad(pred.sum(), [atom_in, bond_in])
            except:
                print('WARNING: grad error!')
                atom_grad = torch.zeros_like(atom_in)
                bond_grad = torch.zeros_like(bond_in)

        # multiply mask, std
        atom_grad = atom_grad * kwargs['atom_mask'].unsqueeze(-1)
        bond_grad = bond_grad * kwargs['bond_mask']

        if only_grad:
            if std:
                return atom_grad, bond_grad, atom_std, bond_std
            return atom_grad, bond_grad

        atom_norm = torch.norm(atom_grad.reshape(atom_grad.shape[0], -1), dim=-1)
        bond_norm = torch.norm(bond_grad.reshape(bond_grad.shape[0], -1), dim=-1)

        if norm:
            atom_grad = atom_grad / (atom_norm + 1e-8)[:, None, None]
            bond_grad = bond_grad / (bond_norm + 1e-8)[:, None, None, None]

        atom_grad = - atom_std[:, None, None] * atom_grad
        bond_grad = - bond_std[:, None, None, None] * bond_grad
        return atom_grad, bond_grad

    return mol_regressor_grad_fn


def get_guided_theta_fn(theta_fn, regressor_grad_fn, guidance_scale=1.0):
    """theta function with gradient guidance."""
    def guided_theta_fn(x, t, *args, **kwargs):
        atom_theta, bond_theta = theta_fn(x, t, *args, **kwargs)
        atom_grad, bond_grad = regressor_grad_fn(x, t, *args, **kwargs)

        # atom_grad, bond_grad, atom_norm, bond_norm, atom_std, bond_std = regressor_grad_fn(x, t, True, *args, **kwargs)
        # atom_score = - atom_theta / atom_std[:, None, None]
        # atom_score_norm = torch.norm(atom_score.reshape(atom_score.shape[0], -1), dim=-1)
        # bond_score = - bond_theta / bond_std[:, None, None, None]
        # bond_score_norm = torch.norm(bond_score.reshape(bond_score.shape[0], -1), dim=-1)
        # atom_grad = - atom_std[:, None, None] * atom_grad * atom_score_norm[:, None, None] / (atom_norm + 1e-8)[:, None, None]
        # bond_grad = - bond_std[:, None, None, None] * bond_grad * bond_score_norm[:, None, None, None] / (bond_norm + 1e-8)[:, None, None, None]

        return atom_theta + atom_grad * guidance_scale, bond_theta + bond_grad * guidance_scale

    return guided_theta_fn


def get_theta_fn(sde, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def theta_fn(x, t, *args, **kwargs):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for continuously-trained models.
                labels = t * 999
                theta = model_fn(x, labels, *args, **kwargs)
            else:
                raise NotImplementedError()
            return theta
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return theta_fn

@torch.no_grad()
def get_rw_feat(k_step, dense_adj):
    """Compute k_step Random Walk for given dense adjacency matrix."""

    rw_list = []
    deg = dense_adj.sum(-1, keepdims=True)
    AD = dense_adj / (deg + 1e-8)
    rw_list.append(AD)

    for _ in range(k_step):
        rw = torch.bmm(rw_list[-1], AD)
        rw_list.append(rw)
    rw_map = torch.stack(rw_list[1:], dim=1)  # [B, k_step, N, N]

    rw_landing = torch.diagonal(rw_map, offset=0, dim1=2, dim2=3)  # [B, k_step, N]
    rw_landing = rw_landing.permute(0, 2, 1)  # [B, N, rw_depth]

    # get the shortest path distance indices
    tmp_rw = rw_map.sort(dim=1)[0]
    spd_ind = (tmp_rw <= 0).sum(dim=1)  # [B, N, N]

    spd_onehot = torch.nn.functional.one_hot(spd_ind, num_classes=k_step+1).to(torch.float)
    spd_onehot = spd_onehot.permute(0, 3, 1, 2)  # [B, kstep, N, N]

    return rw_landing, spd_onehot
