# DPM solvers: stiff semi-linear ODE
# Note: hyperparams of Atom_SDE and Bond_SDE should keep the same for DPM-Solver-1, DPM-Solver-2 and DPM-Solver-3 !!!

import torch
import numpy as np
import functools

from models.utils import get_multi_theta_fn, get_multi_score_fn, get_theta_fn


def sample_nodes(n_nodes_pmf, atom_shape, device):
    n_nodes = torch.multinomial(n_nodes_pmf, atom_shape[0], replacement=True)
    atom_mask = torch.zeros((atom_shape[0], atom_shape[1]), device=device)
    for i in range(atom_shape[0]):
        atom_mask[i][:n_nodes[i]] = 1.
    bond_mask = (atom_mask[:, None, :] * atom_mask[:, :, None]).unsqueeze(1)
    bond_mask = torch.tril(bond_mask, -1)
    bond_mask = bond_mask + bond_mask.transpose(-1, -2)
    return n_nodes, atom_mask, bond_mask


def expand_dim(x, n_dim):
    if n_dim == 3:
        x = x[:, None, None]
    elif n_dim == 4:
        x = x[:, None, None, None]
    return x


def dpm1_update(x_last, t_last, t_i, sde, theta):
    # dpm_solver 1 order update function
    expand_fn = functools.partial(expand_dim, n_dim=len(x_last.shape))

    lambda_i, alpha_i, std_i = sde.log_snr(t_i)
    lambda_last, alpha_last, _ = sde.log_snr(t_last)
    h_i = lambda_i - lambda_last

    x_i = expand_fn(alpha_i / alpha_last) * x_last - expand_fn(std_i * torch.expm1(h_i)) * theta
    return x_i


def dpm_mol_solver_1(atom_sde, bond_sde, theta_fn, x_atom_last, x_bond_last,
                     t_last, t_i, atom_mask, bond_mask):
    # run solver func once

    vec_t_last = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_last
    vec_t_i = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_i
    atom_fn = functools.partial(expand_dim, n_dim=len(x_atom_last.shape))
    bond_fn = functools.partial(expand_dim, n_dim=len(x_bond_last.shape))

    lambda_i, alpha_i, std_i = atom_sde.log_snr(vec_t_i)
    lambda_last, alpha_last, _ = atom_sde.log_snr(vec_t_last)
    h_i = lambda_i - lambda_last

    atom_theta, bond_theta = theta_fn((x_atom_last, x_bond_last), vec_t_last, atom_mask=atom_mask, bond_mask=bond_mask)
    tmp_linear = alpha_i / alpha_last
    tmp_nonlinear = std_i * torch.expm1(h_i)
    x_atom_i = atom_fn(tmp_linear) * x_atom_last - atom_fn(tmp_nonlinear) * atom_theta
    x_bond_i = bond_fn(tmp_linear) * x_bond_last - bond_fn(tmp_nonlinear) * bond_theta

    return x_atom_i, x_bond_i


def dpm_mol_solver_2(atom_sde, bond_sde, theta_fn, x_atom_last, x_bond_last,
                     t_last, t_i, atom_mask, bond_mask, r1=0.5):
    vec_t_last = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_last
    vec_t_i = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_i
    atom_fn = functools.partial(expand_dim, n_dim=len(x_atom_last.shape))
    bond_fn = functools.partial(expand_dim, n_dim=len(x_bond_last.shape))

    lambda_i, alpha_i, std_i = atom_sde.log_snr(vec_t_i)
    lambda_last, alpha_last, _ = atom_sde.log_snr(vec_t_last)
    h_i = lambda_i - lambda_last

    s_i = atom_sde.lambda2t(lambda_last + r1 * h_i)
    _, alpha_si, std_si = atom_sde.log_snr(s_i)
    atom_theta_0, bond_theta_0 = theta_fn((x_atom_last, x_bond_last), vec_t_last,
                                          atom_mask=atom_mask, bond_mask=bond_mask)

    tmp_lin = alpha_si / alpha_last
    tmp_nonlin = std_si * torch.expm1(r1 * h_i)
    u_atom_i = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin) * atom_theta_0
    u_bond_i = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin) * bond_theta_0

    atom_theta_si, bond_theta_si = theta_fn((u_atom_i, u_bond_i), s_i, atom_mask=atom_mask, bond_mask=bond_mask)

    tmp_lin = alpha_i / alpha_last
    tmp_nonlin1 = std_i * torch.expm1(h_i)
    tmp_nonlin2 = (std_i / (2. * r1)) * torch.expm1(h_i)
    x_atom_i = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin1) * atom_theta_0 - \
               atom_fn(tmp_nonlin2) * (atom_theta_si - atom_theta_0)
    x_bond_i = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin1) * bond_theta_0 - \
               bond_fn(tmp_nonlin2) * (bond_theta_si - bond_theta_0)

    return x_atom_i, x_bond_i


def dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom_last, x_bond_last,
                     t_last, t_i, atom_mask, bond_mask, r1=1./3., r2=2./3.):
    vec_t_last = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_last
    vec_t_i = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_i
    atom_fn = functools.partial(expand_dim, n_dim=len(x_atom_last.shape))
    bond_fn = functools.partial(expand_dim, n_dim=len(x_bond_last.shape))

    lambda_i, alpha_i, std_i = atom_sde.log_snr(vec_t_i)
    lambda_last, alpha_last, _ = atom_sde.log_snr(vec_t_last)
    h_i = lambda_i - lambda_last

    s1 = atom_sde.lambda2t(lambda_last + r1 * h_i)
    s2 = atom_sde.lambda2t(lambda_last + r2 * h_i)

    _, alpha_s1, std_s1 = atom_sde.log_snr(s1)
    _, alpha_s2, std_s2 = atom_sde.log_snr(s2)

    atom_theta_0, bond_theta_0 = theta_fn((x_atom_last, x_bond_last), vec_t_last,
                                          atom_mask=atom_mask, bond_mask=bond_mask)

    tmp_lin = alpha_s1 / alpha_last
    tmp_nonlin = std_s1 * torch.expm1(r1 * h_i)
    u_atom_1 = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin) * atom_theta_0
    u_bond_1 = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin) * bond_theta_0

    atom_theta_s1, bond_theta_s1 = theta_fn((u_atom_1, u_bond_1), s1, atom_mask=atom_mask, bond_mask=bond_mask)
    D_atom_1 = atom_theta_s1 - atom_theta_0
    D_bond_1 = bond_theta_s1 - bond_theta_0

    tmp_lin = alpha_s2 / alpha_last
    tmp_nonlin1 = std_s2 * torch.expm1(r2 * h_i)
    tmp_nonlin2 = (std_s2 * r2 / r1) * (torch.expm1(r2 * h_i) / (r2 * h_i) - 1)
    u_atom_2 = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin1) * atom_theta_0 - atom_fn(tmp_nonlin2) * D_atom_1
    u_bond_2 = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin1) * bond_theta_0 - bond_fn(tmp_nonlin2) * D_bond_1

    atom_theta_s2, bond_theta_s2 = theta_fn((u_atom_2, u_bond_2), s2, atom_mask=atom_mask, bond_mask=bond_mask)
    D_atom_2 = atom_theta_s2 - atom_theta_0
    D_bond_2 = bond_theta_s2 - bond_theta_0

    tmp_lin = alpha_i / alpha_last
    tmp_nonlin1 = std_i * torch.expm1(h_i)
    tmp_nonlin2 = (std_i / r2) * (torch.expm1(h_i) / h_i - 1)
    x_atom_i = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin1) * atom_theta_0 - atom_fn(tmp_nonlin2) * D_atom_2
    x_bond_i = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin1) * bond_theta_0 - bond_fn(tmp_nonlin2) * D_bond_2

    return x_atom_i, x_bond_i


def dpm_solver_3(sde, theta_fn, x_last, t_last, t_i, mask, r1=1./3., r2=2./3.):
    vec_t_last = torch.ones(x_last.shape[0], device=x_last.device) * t_last
    vec_t_i = torch.ones(x_last.shape[0], device=x_last.device) * t_i
    expand_fn = functools.partial(expand_dim, n_dim=len(x_last.shape))

    lambda_i, alpha_i, std_i = sde.log_snr(vec_t_i)
    lambda_last, alpha_last, _ = sde.log_snr(vec_t_last)
    h_i = lambda_i - lambda_last

    s1 = sde.lambda2t(lambda_last + r1 * h_i)
    s2 = sde.lambda2t(lambda_last + r2 * h_i)

    _, alpha_s1, std_s1 = sde.log_snr(s1)
    _, alpha_s2, std_s2 = sde.log_snr(s2)

    theta_0 = theta_fn(x_last, vec_t_last, mask=mask)

    tmp_lin = alpha_s1 / alpha_last
    tmp_nonlin = std_s1 * torch.expm1(r1 * h_i)
    u_1 = expand_fn(tmp_lin) * x_last - expand_fn(tmp_nonlin) * theta_0

    theta_s1 = theta_fn(u_1, s1, mask=mask)
    D_1 = theta_s1 - theta_0

    tmp_lin = alpha_s2 / alpha_last
    tmp_nonlin1 = std_s2 * torch.expm1(r2 * h_i)
    tmp_nonlin2 = (std_s2 * r2 / r1) * (torch.expm1(r2 * h_i) / (r2 * h_i) - 1)
    u_2 = expand_fn(tmp_lin) * x_last - expand_fn(tmp_nonlin1) * theta_0 - expand_fn(tmp_nonlin2) * D_1

    theta_s2 = theta_fn(u_2, s2, mask=mask)
    D_2 = theta_s2 - theta_0

    tmp_lin = alpha_i / alpha_last
    tmp_nonlin1 = std_i * torch.expm1(h_i)
    tmp_nonlin2 = (std_i / r2) * (torch.expm1(h_i) / h_i - 1)
    x_i = expand_fn(tmp_lin) * x_last - expand_fn(tmp_nonlin1) * theta_0 - expand_fn(tmp_nonlin2) * D_2

    return x_i


def get_mol_sampler_dpm1(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                         time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    start_lambda = atom_sde.log_snr_np(atom_sde.T)
    stop_lambda = atom_sde.log_snr_np(eps)
    lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=int(time_step + 1))
    time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

    # time_steps = np.linspace(start=atom_sde.T, stop=eps, num=int(time_step + 1))

    def sampler(model, n_nodes_pmf, z=None):
        with torch.no_grad():
            # set up dpm theta func
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # initial sample
            assert z is None
            # If not represent, sample the latent code from the prior distribution of the SDE.
            x_atom = atom_sde.prior_sampling(atom_shape).to(device)
            x_bond = bond_sde.prior_sampling(bond_shape).to(device)

            # Sample the number of nodes, if z is None
            n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
            x_atom = x_atom * atom_mask.unsqueeze(-1)
            x_bond = x_bond * bond_mask

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
                x_atom, x_bond = dpm_mol_solver_1(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
                t_last = t_i

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, len(time_steps) - 1, n_nodes

    return sampler


def get_mol_sampler_dpm2(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                         time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 2)

    start_lambda = atom_sde.log_snr_np(atom_sde.T)
    stop_lambda = atom_sde.log_snr_np(eps)
    lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step+1)
    time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

    # time_steps = np.linspace(start=atom_sde.T, stop=eps, num=num_step + 1)

    def sampler(model, n_nodes_pmf, z=None):
        with torch.no_grad():
            # set up dpm theta func
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # initial sample
            assert z is None
            # If not represent, sample the latent code from the prior distribution of the SDE.
            x_atom = atom_sde.prior_sampling(atom_shape).to(device)
            x_bond = bond_sde.prior_sampling(bond_shape).to(device)

            # Sample the number of nodes, if z is None
            n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
            x_atom = x_atom * atom_mask.unsqueeze(-1)
            x_bond = x_bond * bond_mask

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
                x_atom, x_bond = dpm_mol_solver_2(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
                t_last = t_i

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, num_step * 2, n_nodes

    return sampler


def get_mol_sampler_dpm3(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                         time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, n_nodes_pmf=None, time_point=None, z=None, atom_mask=None, bond_mask=None, theta_fn=None):
        if time_point is None:
            start_lambda = atom_sde.log_snr_np(atom_sde.T)
            stop_lambda = atom_sde.log_snr_np(eps)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]
        else:
            start_time, stop_time = time_point
            start_lambda = atom_sde.log_snr_np(start_time)
            stop_lambda = atom_sde.log_snr_np(stop_time)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func
            if theta_fn is None:
                theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)
            else:
                theta_fn = theta_fn

            # initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x_atom = atom_sde.prior_sampling(atom_shape).to(device)
                x_bond = bond_sde.prior_sampling(bond_shape).to(device)

                # Sample the number of nodes, if z is None
                n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
                x_atom = x_atom * atom_mask.unsqueeze(-1)
                x_bond = x_bond * bond_mask
            else:
                # just use the concurrent prior z and node_mask, bond_mask
                x_atom, x_bond = z
                n_nodes = atom_mask.sum(-1).long()

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
                x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
                t_last = t_i

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, num_step * 3, n_nodes

    return sampler


def get_mol_encoder_dpm3(atom_sde, bond_sde, time_step, eps=1e-3, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, batch, time_point=None):
        if time_point is None:
            start_lambda = atom_sde.log_snr_np(atom_sde.T)
            stop_lambda = atom_sde.log_snr_np(eps)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]
            time_steps.reverse()
        else:
            start_time, stop_time = time_point
            start_lambda = atom_sde.log_snr_np(start_time)
            stop_lambda = atom_sde.log_snr_np(stop_time)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # run forward deterministic diffusion process
            x_atom, atom_mask, x_bond, bond_mask = batch

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
                x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
                # pdb.set_trace()
                t_last = t_i

            return x_atom, x_bond, num_step * 3

    return sampler


def get_mol_sampler_dpm_mix(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                            time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    start_lambda = atom_sde.log_snr_np(atom_sde.T)
    stop_lambda = atom_sde.log_snr_np(eps)
    lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step+1)
    time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

    R = int(time_step) % 3
    # time_steps = np.linspace(start=atom_sde.T, stop=eps, num=num_step + 1)

    def sampler(model, n_nodes_pmf, z=None):
        with torch.no_grad():
            # set up dpm theta func
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # initial sample
            assert z is None
            # If not represent, sample the latent code from the prior distribution of the SDE.
            x_atom = atom_sde.prior_sampling(atom_shape).to(device)
            x_bond = bond_sde.prior_sampling(bond_shape).to(device)

            # Sample the number of nodes, if z is None
            n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
            x_atom = x_atom * atom_mask.unsqueeze(-1)
            x_bond = x_bond * bond_mask

            # run solver func according to time schedule
            t_last = time_steps[0]

            if R == 0:
                for t_i in time_steps[1:-2]:
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                    t_last = t_i
                t_i = time_steps[-2]
                x_atom, x_bond = dpm_mol_solver_2(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
                t_last = t_i
                t_i = time_steps[-1]
                x_atom, x_bond = dpm_mol_solver_1(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
            else:
                for t_i in time_steps[1:-1]:
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                    t_last = t_i
                t_i = time_steps[-1]
                if R == 1:
                    x_atom, x_bond = dpm_mol_solver_1(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                elif R == 2:
                    x_atom, x_bond = dpm_mol_solver_2(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                else:
                    raise ValueError('Step Error in mix DPM-solver.')

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, time_step, n_nodes

    return sampler


def get_sampler_dpm3(sde, shape, inverse_scaler, time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, n_nodes_pmf=None, time_point=None, z=None, mask=None, theta_fn=None):
        if time_point is None:
            start_lambda = sde.log_snr_np(sde.T)
            stop_lambda = sde.log_snr_np(eps)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]
        else:
            start_time, stop_time = time_point
            start_lambda = sde.log_snr_np(start_time)
            stop_lambda = sde.log_snr_np(stop_time)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func
            if theta_fn is None:
                theta_fn = get_theta_fn(sde, model, train=False, continuous=True)
            else:
                theta_fn = theta_fn

            # initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x = sde.prior_sampling(shape).to(device)
                # Sample the number of nodes, if z is None
                n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
                mask = torch.zeros((shape[0], shape[-1]), device=device)
                for i in range(shape[0]):
                    mask[i][:n_nodes[i]] = 1.
                mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)

            else:
                x = z
                batch_size, _, max_num_nodes, _ = mask.shape
                node_mask = mask[:, 0, 0, :].clone()  # without checking correctness
                node_mask[:, 0] = 1
                n_nodes = node_mask.sum(-1).long()

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
                x = dpm_solver_3(sde, theta_fn, x, t_last, t_i, mask)
                t_last = t_i

            if denoise:
                pass

            x = inverse_scaler(x) * mask
            return x, num_step * 3, n_nodes

    return sampler


def get_mol_dpm3_twostage(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                          time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, n_nodes_pmf, time_point, guided_theta_fn):

        start_lambda = atom_sde.log_snr_np(atom_sde.T)
        stop_lambda = atom_sde.log_snr_np(eps)
        lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
        time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # initial sample
            x_atom = atom_sde.prior_sampling(atom_shape).to(device)
            x_bond = bond_sde.prior_sampling(bond_shape).to(device)

            # Sample the number of nodes, if z is None
            n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
            x_atom = x_atom * atom_mask.unsqueeze(-1)
            x_bond = x_bond * bond_mask

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
                if t_last > time_point:
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                else:
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, guided_theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                t_last = t_i

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, num_step * 3, n_nodes

    return sampler
