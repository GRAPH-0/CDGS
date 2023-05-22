Conditional Diffusion Based on Discrete Graph Structures for Molecular Graph Generation - AAAI 2023

## Dependencies

* pytorch 1.11
* PyG 2.1

For NSPDK evaluation:

```pip install git+https://github.com/fabriziocosta/EDeN.git --user```

Others see requirements.txt .


## Training 

### QM9

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/vp_qm9_cdgs.py --mode train --workdir exp/vpsde_qm9_cdgs
``` 

* Set GPU id via `CUDA_VISIBLE_DEVICES`.
* `workdir` is the directory path to save checkpoints, which can be changed to `YOUR_PATH`. We provide the pretrained checkpoint in `exp/vpsde_qm9_cdgs`.
* More hyperparameters in the config file `configs/vp_qm9_cdgs.py`

### ZINC250k

## Sampling

### QM9

1. EM sampling with 1000 steps

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/vp_qm9_cdgs.py --mode eval --workdir exp/vpsde_qm9_cdgs --config.eval.begin_ckpt 200 --config.eval.end_ckpt 200
```

* Add `--config.eval.nspdk` if apply NSPDK evaluation.
* Change iteration steps through `--config.model.num_scales YOUR_STEPS`.

## Citation

```bibtex
@article{huang2023conditional,
  title={Conditional Diffusion Based on Discrete Graph Structures for Molecular Graph Generation},
  author={Huang, Han and Sun, Leilei and Du, Bowen and Lv, Weifeng},
  journal={arXiv preprint arXiv:2301.00427},
  year={2023}
}
```

