<h1 align="center">Discrete Neural Flow Sampler (DNFS)</h1>
<p align="center">
    <a href="https://neurips.cc/virtual/2025/poster/117571"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2725&color=blue"> </a>
    <a href=""> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a>
    <a href=""> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a>
    <a href=""> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a>
</p>

This repo contains PyTorch implementation of the paper "[Discrete Neural Flow Samplers with Locally Equivariant Transformer](https://arxiv.org/abs/2505.17741)"

by [Zijing Ou](https://j-zin.github.io/), [Ruixiang Zhang](https://ruixiangz.me/) and [Yingzhen Li](http://yingzhenli.net/home/en/).

> we propose Discrete Neural Flow Samplers (DNFS), a trainable and efficient framework for discrete sampling. DNFS learns the rate matrix of a continuous-time Markov chain such that the resulting dynamics satisfy the Kolmogorov equation. As this objective involves the intractable partition function, we then employ control variates to reduce the variance of its Monte Carlo estimation, leading to a coordinate descent learning algorithm. To further facilitate computational efficiency, we propose locally equivaraint Transformer, a novel parameterisation of the rate matrix that significantly improves training efficiency while preserving powerful network expressiveness.

## Experiments

We provide a minimum code to reproduce DNFA on sampling from Ising models. To train the model, please run:


```bash
# 5x5 Ising model
python main.py --ising_dim 5 --ising_sigma 0.1 --ising_bias 0.2 --eval_every 1 
# 10x10 Ising model
python main.py --ising_dim 10 --ising_sigma 0.1 --ising_bias 0.0 --eval_every 5
python main.py --ising_dim 10 --ising_sigma 0.22305 --ising_bias 0.0 --eval_every 5
```

## Environment

Local development uses [pixi](https://pixi.sh). After cloning:

```bash
pixi install
pixi run smoke   # 2-epoch CPU sanity check
```

## Running on Modal

Training runs on Modal with an L4 GPU. One-time setup (creates the `wandb` secret Modal needs):

```bash
pixi run modal secret create wandb WANDB_API_KEY=<key-from-https://wandb.ai/authorize>
```

Then:

```bash
# Short sanity run
pixi run modal run modal_app.py --ising-dim 5 --epochs 5

# Full runs
pixi run modal run modal_app.py --ising-dim 5  --ising-sigma 0.1     --ising-bias 0.2 --eval-every 1
pixi run modal run modal_app.py --ising-dim 10 --ising-sigma 0.1     --ising-bias 0.0 --eval-every 5
pixi run modal run modal_app.py --ising-dim 10 --ising-sigma 0.22305 --ising-bias 0.0 --eval-every 5
```
