"""Various sampling methods."""

import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn, get_multi_score_fn
from scipy import integrate
# from torchdiffeq import odeint
import sde_lib
from models import utils as mutils
from dpm_solvers import get_mol_sampler_dpm1, get_mol_sampler_dpm2, get_mol_sampler_dpm3, \
    get_mol_sampler_dpm_mix, get_sampler_dpm3
import time


_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered predictor with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered corrector with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_mol_sampling_fn(config, atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler, eps):
    """Create a sampling function for molecule.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        atom_sde, bond_sde: A `sde_lib.SDE` object that represents the forward SDE.
        atom_shape, bond_shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    if sampler_name.lower() == 'dpm1':
        sampling_fn = get_mol_sampler_dpm1(atom_sde=atom_sde,
                                           bond_sde=bond_sde,
                                           atom_shape=atom_shape,
                                           bond_shape=bond_shape,
                                           inverse_scaler=inverse_scaler,
                                           time_step=config.sampling.ode_step,
                                           eps=eps,
                                           denoise=config.sampling.noise_removal,
                                           device=config.device)
    elif sampler_name.lower() == 'dpm2':
        sampling_fn = get_mol_sampler_dpm2(atom_sde=atom_sde,
                                           bond_sde=bond_sde,
                                           atom_shape=atom_shape,
                                           bond_shape=bond_shape,
                                           inverse_scaler=inverse_scaler,
                                           time_step=config.sampling.ode_step,
                                           eps=eps,
                                           denoise=config.sampling.noise_removal,
                                           device=config.device)
    elif sampler_name.lower() == 'dpm3':
        sampling_fn = get_mol_sampler_dpm3(atom_sde=atom_sde,
                                           bond_sde=bond_sde,
                                           atom_shape=atom_shape,
                                           bond_shape=bond_shape,
                                           inverse_scaler=inverse_scaler,
                                           time_step=config.sampling.ode_step,
                                           eps=eps,
                                           denoise=config.sampling.noise_removal,
                                           device=config.device)
    elif sampler_name.lower() == 'dpm_mix':
        sampling_fn = get_mol_sampler_dpm_mix(atom_sde=atom_sde,
                                              bond_sde=bond_sde,
                                              atom_shape=atom_shape,
                                              bond_shape=bond_shape,
                                              inverse_scaler=inverse_scaler,
                                              time_step=config.sampling.ode_step,
                                              eps=eps,
                                              denoise=config.sampling.noise_removal,
                                              device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':

        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())

        sampling_fn = get_mol_pc_sampler(atom_sde=atom_sde,
                                         bond_sde=bond_sde,
                                         atom_shape=atom_shape,
                                         bond_shape=bond_shape,
                                         predictor=predictor,
                                         corrector=corrector,
                                         inverse_scaler=inverse_scaler,
                                         snr=(config.sampling.atom_snr, config.sampling.bond_snr),
                                         n_steps=config.sampling.n_steps_each,
                                         probability_flow=config.sampling.probability_flow,
                                         continuous=config.training.continuous,
                                         denoise=config.sampling.noise_removal,
                                         eps=eps,
                                         device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if isinstance(sde, tuple):
            self.rsde = (sde[0].reverse(score_fn, probability_flow), sde[1].reverse(score_fn, probability_flow))
        else:
            self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, *args, **kwargs):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    @abc.abstractmethod
    def update_mol_fn(self, x, t, *args, **kwargs):
        """One update of the predictor for molecule graphs.

        Args:
            x: A tuple of PyTorch tensor (x_atom, x_bond) representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A tuple of PyTorch tensor (x_atom, x_bond) of the next state.
            x_mean: A tuple of PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, *args, **kwargs):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    @abc.abstractmethod
    def update_mol_fn(self, x, t, *args, **kwargs):
        """One update of the corrector for molecule graphs.

        Args:
            x: A tuple of PyTorch tensor (x_atom, x_bond) representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A tuple of PyTorch tensor (x_atom, x_bond) of the next state.
            x_mean: A tuple of PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        z = torch.tril(z, -1)
        z = z + z.transpose(-1, -2)
        drift, diffusion = self.rsde.sde(x, t, *args, **kwargs)
        drift = torch.tril(drift, -1)
        drift = drift + drift.transpose(-1, -2)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean

    def update_mol_fn(self, x, t, *args, **kwargs):
        atom_score, bond_score = self.score_fn(x, t, *args, **kwargs)
        # print('predictor atom norm: ', torch.norm(atom_score.reshape(atom_score.shape[0], -1), dim=-1).mean(), t[0])

        x_atom, x_bond = x
        dt = -1. / self.rsde[0].N

        # atom update
        z_atom = torch.randn_like(x_atom)
        drift_atom, diffusion_atom = self.rsde[0].sde_score(x_atom, t, atom_score)
        x_atom_mean = x_atom + drift_atom * dt
        x_atom = x_atom_mean + diffusion_atom[:, None, None] * np.sqrt(-dt) * z_atom

        # bond update
        z_bond = torch.randn_like(x_bond)
        z_bond = torch.tril(z_bond, -1)
        z_bond = z_bond + z_bond.transpose(-1, -2)
        drift_bond, diffusion_bond = self.rsde[1].sde_score(x_bond, t, bond_score)

        x_bond_mean = x_bond + drift_bond * dt
        x_bond = x_bond_mean + diffusion_bond[:, None, None, None] * np.sqrt(-dt) * z_bond

        return (x_atom, x_bond), (x_atom_mean, x_bond_mean)


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t, *args, **kwargs):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            # Note: it seems that subVPSDE doesn't set alphas
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):

            grad = score_fn(x, t, *args, **kwargs)
            noise = torch.randn_like(x)

            noise = torch.tril(noise, -1)
            noise = noise + noise.transpose(-1, -2)

            mask = kwargs['mask']

            # mask invalid elements and calculate norm
            mask_tmp = mask.reshape(mask.shape[0], -1)

            grad_norm = torch.norm(mask_tmp * grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(mask_tmp * noise.reshape(noise.shape[0], -1), dim=-1).mean()

            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


    def update_mol_fn(self, x, t, *args, **kwargs):
        x_atom, x_bond = x
        atom_sde, bond_sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        atom_snr, bond_snr = self.snr
        if isinstance(atom_sde, sde_lib.VPSDE) or isinstance(atom_sde, sde_lib.subVPSDE):
            timestep = (t * (atom_sde.N - 1) / atom_sde.T).long()
            # Note: it seems that subVPSDE doesn't set alphas
            alpha_atom = atom_sde.alphas.to(t.device)[timestep]
            alpha_bond = bond_sde.alphas.to(t.device)[timestep]
        else:
            alpha_atom = alpha_bond = torch.ones_like(t)

        for i in range(n_steps):
            grad_atom, grad_bond = score_fn(x, t, *args, **kwargs)

            # update atom
            noise_atom = torch.randn_like(x_atom)
            noise_atom = noise_atom * kwargs['atom_mask'].unsqueeze(-1)

            ## mask invalid elements and calculate norm
            # atom_mask = kwargs['atom_mask'].unsqueeze(-1)
            # atom_mask = atom_mask.repeat(1, 1, grad_atom.shape[-1]).reshape(grad_atom.shape[0], -1)

            # grad_norm_a = torch.norm(atom_mask * grad_atom.reshape(grad_atom.shape[0], -1), dim=-1).mean()
            # noise_norm_a = torch.norm(atom_mask * noise_atom.reshape(noise_atom.shape[0], -1), dim=-1).mean()
            grad_norm_a = torch.norm(grad_atom.reshape(grad_atom.shape[0], -1), dim=-1).mean()
            noise_norm_a = torch.norm(noise_atom.reshape(noise_atom.shape[0], -1), dim=-1).mean()

            # print('Corrector atom score norm:', grad_norm_a, t[0])
            step_size_a = (atom_snr * noise_norm_a / grad_norm_a) ** 2 * 2 * alpha_atom
            x_atom_mean = x_atom + step_size_a[:, None, None] * grad_norm_a
            x_atom = x_atom_mean + torch.sqrt(step_size_a * 2)[:, None, None] * noise_atom

            # update bond
            noise_bond = torch.randn_like(x_bond)
            noise_bond = torch.tril(noise_bond, -1)
            noise_bond = noise_bond + noise_bond.transpose(-1, -2)
            noise_bond = noise_bond * kwargs['bond_mask']

            # bond_mask = kwargs['bond_mask'].repeat(1, grad_bond.shape[1], 1, 1).reshape(grad_bond.shape[0], -1)
            # grad_norm_b = torch.norm(bond_mask * grad_bond.reshape(grad_bond.shape[0], -1), dim=-1).mean()
            # noise_norm_b = torch.norm(bond_mask * noise_bond.reshape(noise_bond.shape[0], -1), dim=-1).mean()
            grad_norm_b = torch.norm(grad_bond.reshape(grad_bond.shape[0], -1), dim=-1).mean()
            noise_norm_b = torch.norm(noise_bond.reshape(noise_bond.shape[0], -1), dim=-1).mean()

            step_size_b = (bond_snr * noise_norm_b / grad_norm_b) ** 2 * 2 * alpha_bond
            x_bond_mean = x_bond + step_size_b[:, None, None, None] * grad_norm_b
            x_bond = x_bond_mean + torch.sqrt(step_size_b * 2)[:, None, None, None] * noise_bond

        return (x_atom, x_bond), (x_atom_mean, x_bond_mean)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x

    def update_mol_fn(self, x, t, *args, **kwargs):
        return x, x


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x

    def update_atom_fn(self, x, t, *args, **kwargs):
        return x, x

    def update_bond_fn(self, x, t, *args, **kwargs):
        return x, x

    def update_mol_fn(self, x, t, *args, **kwargs):
        return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous, *args, **kwargs):
    """A wrapper that configures and returns the update function of predictors."""
    if isinstance(sde, tuple):
        score_fn = mutils.get_multi_score_fn(sde[0], sde[1], model, train=False, continuous=continuous)
    else:
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    if isinstance(sde, tuple):
        return predictor_obj.update_mol_fn(x, t, *args, **kwargs)
    return predictor_obj.update_fn(x, t, *args, **kwargs)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps, *args, **kwargs):
    """A wrapper that configures and returns the update function of correctors."""
    if isinstance(sde, tuple):
        score_fn = mutils.get_multi_score_fn(sde[0], sde[1], model, train=False, continuous=continuous)
    else:
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    if isinstance(sde, tuple):
        return corrector_obj.update_mol_fn(x, t, *args, **kwargs)
    return corrector_obj.update_fn(x, t, *args, **kwargs)


def get_mol_pc_sampler(atom_sde, bond_sde, atom_shape, bond_shape, predictor, corrector, inverse_scaler, snr,
                       n_steps=1, probability_flow=False, continuous=False,
                       denoise=True, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler for molecule graph generation.

    Args:
        atom_sde, bond_sde: An `sde_lib.SDE` object representing the forward SDE.
        atom_shape, bond_shape: A sequence of integers. The expected shape of a single sample.
        predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
        corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
        inverse_scaler: The inverse data normalizer.
        snr: A `float` number. The signal-to-noise ratio for configuring correctors.
        n_steps: An integer. The number of corrector steps per predictor update.
        probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
        continuous: `True` indicates that the score model was continuously trained.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=(atom_sde, bond_sde),
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=(atom_sde, bond_sde),
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)


    def mol_pc_sampler(model, n_nodes_pmf):
        """The PC sampler function.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.

        Returns:
            Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample

            x_atom = atom_sde.prior_sampling(atom_shape).to(device)
            x_bond = bond_sde.prior_sampling(bond_shape).to(device)

            timesteps = torch.linspace(atom_sde.T, eps, atom_sde.N, device=device)

            # Sample the number of nodes
            n_nodes = torch.multinomial(n_nodes_pmf, atom_shape[0], replacement=True)
            atom_mask = torch.zeros((atom_shape[0], atom_shape[1]), device=device)
            for i in range(atom_shape[0]):
                atom_mask[i][:n_nodes[i]] = 1.
            bond_mask = (atom_mask[:, None, :] * atom_mask[:, :, None]).unsqueeze(1)
            bond_mask = torch.tril(bond_mask, -1)
            bond_mask = bond_mask + bond_mask.transpose(-1, -2)

            x_atom = x_atom * atom_mask.unsqueeze(-1)
            x_bond = x_bond * bond_mask

            for i in range(atom_sde.N):
                t = timesteps[i]
                vec_t = torch.ones(atom_shape[0], device=t.device) * t

                (x_atom, x_bond), (x_atom_mean, x_bond_mean) = corrector_update_fn((x_atom, x_bond), vec_t, model=model,
                                                                                   atom_mask=atom_mask,
                                                                                   bond_mask=bond_mask)
                x_atom = x_atom * atom_mask.unsqueeze(-1)
                x_bond = x_bond * bond_mask

                (x_atom, x_bond), (x_atom_mean, x_bond_mean) = predictor_update_fn((x_atom, x_bond), vec_t, model=model,
                                                                                   atom_mask=atom_mask,
                                                                                   bond_mask=bond_mask)
                x_atom = x_atom * atom_mask.unsqueeze(-1)
                x_bond = x_bond * bond_mask

            return inverse_scaler(x_atom_mean if denoise else x_atom, atom=True) * atom_mask.unsqueeze(-1),\
                   inverse_scaler(x_bond_mean if denoise else x_bond, atom=False) * bond_mask,\
                   atom_sde.N * (n_steps + 1), n_nodes

    return mol_pc_sampler
