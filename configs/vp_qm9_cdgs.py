"""Training GNN on QM9 with continuous VPSDE."""

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.model_type = 'mol_sde'

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vpsde'
    training.continuous = True
    training.reduce_mean = False

    training.batch_size = 128
    training.eval_batch_size = 512
    training.n_iters = 1000000
    training.snapshot_freq = 5000  # SET Larger values to save less checkpoints
    training.log_freq = 200
    training.eval_freq = 5000
    ## store additional checkpoints for preemption
    training.snapshot_freq_for_preemption = 2000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'
    sampling.rtol = 1e-5
    sampling.atol = 1e-5
    sampling.ode_method = 'rk4'
    sampling.ode_step = 0.01

    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.atom_snr = 0.16
    sampling.bond_snr = 0.16
    sampling.vis_row = 4
    sampling.vis_col = 4

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 15
    evaluate.end_ckpt = 40
    evaluate.batch_size = 10000  # 1024
    evaluate.enable_sampling = True
    evaluate.num_samples = 10000
    evaluate.mmd_distance = 'RBF'
    evaluate.max_subgraph = False
    evaluate.save_graph = False
    evaluate.nn_eval = False
    evaluate.nspdk = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = 'data'
    data.name = 'QM9'
    data.split_ratio = 0.8
    data.max_node = 9
    data.atom_channels = 4
    data.bond_channels = 2
    data.atom_list = [6, 7, 8, 9]
    data.norm = (0.5, 1.0)

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'CDGS'
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 64
    model.num_gnn_layers = 6
    model.conditional = True
    model.embedding_type = 'positional'
    model.rw_depth = 8
    model.graph_layer = 'GINE'
    model.edge_th = -1.
    model.heads = 8
    model.dropout = 0.1

    model.num_scales = 1000  # SDE total steps (N)
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.node_beta_min = 0.1
    model.node_beta_max = 20.
    model.edge_beta_min = 0.1
    model.edge_beta_max = 20.

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.  # SET Larger values to converge faster, e.g., 10.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
