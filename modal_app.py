"""Modal entry point for DNFS training on an L4 GPU.

Usage:
    modal run modal_app.py --ising-dim 5 --ising-sigma 0.1 --ising-bias 0.2
    modal run modal_app.py --ising-dim 10 --ising-sigma 0.22305 --ising-bias 0.0 --eval-every 5

Requires a Modal secret named "wandb" exposing WANDB_API_KEY.
"""
from types import SimpleNamespace

import modal

app = modal.App("dnfs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "wandb",
        "tqdm",
        "einops",
        "igraph",
        "numpy",
        "matplotlib",
        "scipy",
    )
    .add_local_python_source(
        "main", "model", "ising_models", "ising_theory", "kolmo_utils"
    )
)

wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=image,
    gpu="L4",
    secrets=[wandb_secret],
    timeout=6 * 60 * 60,
)
def train(
    ising_dim: int,
    ising_sigma: float,
    ising_bias: float,
    eval_every: int,
    epochs: int,
) -> None:
    from main import main

    args = SimpleNamespace(
        ising_dim=ising_dim,
        ising_sigma=ising_sigma,
        ising_bias=ising_bias,
        eval_every=eval_every,
        epochs=epochs,
        device="cuda",
        vocab_size=2,
        T=64,
        N=256,
        lr=1e-3,
        batch_size=128,
        steps_per_epoch=100,
    )
    main(args)


@app.local_entrypoint()
def launch(
    ising_dim: int = 5,
    ising_sigma: float = 0.1,
    ising_bias: float = 0.2,
    eval_every: int = 1,
    epochs: int = 2000,
) -> None:
    train.remote(
        ising_dim=ising_dim,
        ising_sigma=ising_sigma,
        ising_bias=ising_bias,
        eval_every=eval_every,
        epochs=epochs,
    )
