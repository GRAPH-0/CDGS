from fcd_torch import FCD


def compute_intermediate_FCD(smiles, n_jobs=1, device='cpu', batch_size=512):
    """
    Precomputes statistics such as mean and variance for FCD.
    """
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    stats = FCD(**kwargs_fcd).precalc(smiles)
    return stats


def get_FCDMetric(ref_smiles, n_jobs=1, device='cpu', batch_size=512):
    pref = compute_intermediate_FCD(ref_smiles, n_jobs, device, batch_size)

    def FCDMetric(gen_smiles):
        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        return FCD(**kwargs_fcd)(gen=gen_smiles, pref=pref)

    return FCDMetric
