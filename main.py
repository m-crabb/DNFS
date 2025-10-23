import wandb
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from types import SimpleNamespace

from model import LEquiTFRtModel
from ising_models import load_ising_models
from kolmo_utils import UniformInitialDist, generate_train_data_using_rate_matrix, generate_samples_with_importance_weights


def compute_loss(model, xs, ts, dt_log_Zt, time_derivative_log_density, target_density):
    B, D = xs.shape
    dt_log_unormalised_density = time_derivative_log_density(xs, ts)    # (b)

    G_t = model(xs, ts) # (b d s)
    G_t = G_t.scatter_(-1, xs[..., None], 0.0)
    G_t_plus = F.relu(G_t)
    neg_G_t_plus = F.relu(-G_t)

    log_ratio = target_density.get_diff_log_prob(xs, ts)
    log_ratio = log_ratio.scatter_(-1, xs[..., None], 0.0)  # (b d s)
    log_ratio.clamp_(max=5)   # avoid numerical instability

    stein_eq = (G_t_plus - neg_G_t_plus * log_ratio.exp()).reshape(B, -1).sum(dim=-1)    # (b)
    dt_log_density = dt_log_unormalised_density - dt_log_Zt    # (b)

    eps = (dt_log_density + stein_eq).nan_to_num_(posinf=1.0, neginf=-1.0, nan=0.0)
    loss = (eps**2).mean()

    return loss

def main(args):
    wandb.init(
        project="dnfs",
        name=f"ising_dim={args.ising_dim}_sigma={args.ising_sigma}_bias={args.ising_bias}",
        config=args,
        mode="online",
    )

    target_density = load_ising_models(args)
    initial_density = UniformInitialDist(
        D = args.discrete_dim,
        S = args.vocab_size,
        device = args.device
    )

    def sample_initial(num_samples):
        return initial_density.sample(num_samples)
    def time_derivative_log_density(x, t):
        return - initial_density.log_prob(x) + target_density.log_prob(x)

    model = LEquiTFRtModel(
        data_dim=args.discrete_dim,
        vocab_size=args.vocab_size,
        hidden_dim=64,
        head_dim=16,
        num_blocks=3,
        layers_per_block=1,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    current_step = 0
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        model.eval()
        ts = torch.linspace(0, 1, args.T)
        dataset_iterator = generate_train_data_using_rate_matrix(
            model,
            args.N,
            ts,
            sample_initial,
            time_derivative_log_density,
            target_density,
            args,
        )

        model.train()
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            ts_, data, dlogZt = next(dataset_iterator)
            ts_ = ts_.to(args.device)
            data = data.to(args.device)
            dlogZt = dlogZt.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(
                model, data, ts_, dlogZt,
                time_derivative_log_density=time_derivative_log_density,
                target_density=target_density,
            )
            loss.backward()
            optimizer.step()

            current_step += 1
            epoch_loss += loss.item()
            wandb.log({"local/epoch": epoch, "local/loss": loss.item()}, step=current_step)

        avg_loss = epoch_loss / args.steps_per_epoch
        pbar.set_description(f"Epoch {epoch}, Average Loss: {avg_loss}")
        wandb.log(
            {"global/epoch": epoch, "global/average_loss": avg_loss},
            step=current_step
        )
        if epoch % args.eval_every == 0:
            model.eval()
            linear_ts = torch.linspace(0, 1, args.T)

            samples, weights = generate_samples_with_importance_weights(
                model, 2048, 
                sample_initial, time_derivative_log_density, target_density, 
                linear_ts, args
            )

            evaluate(samples, weights, target_density, current_step, args)

def evaluate(samples, weights, target_density, current_step, args):
    # you can run `python ising_theory.py` to get the theorectal values

    num_samples = samples.size(0)
    normalised_weights = F.softmax(weights, dim=0) # normalise weights
    ess_val = 1.0 / torch.sum(torch.exp(2 * normalised_weights.log()))
    ess = (ess_val / num_samples).item()

    beta = 2 * args.ising_sigma
    logZ = weights.mean().item()
    free_energy = -logZ / beta / (args.ising_dim**2)

    log_prob = target_density.log_prob(samples)
    entropy = -(normalised_weights * log_prob).sum()
    internal_energy = entropy / beta / (args.ising_dim**2)

    entropy = (internal_energy - free_energy) / (1 / beta)

    wandb.log(
        {
            "metric/ess": ess,
            "metric/free_energy": free_energy,
            "metric/internal_energy": internal_energy.item(),
            "metric/entropy": entropy.item(),
        },
        step=current_step
    )

    if args.ising_dim == 5:
        gt_mean = args.gt_mean.to(args.device)
        samples = (2 * samples - 1).float()
        log_rmse = ((samples.mean(0) - gt_mean)**2).mean().sqrt().log().item()
        wandb.log(
            {"metric/log_rmse": log_rmse},
            step=current_step
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ising_dim', type=int, default=5)
    parser.add_argument('--ising_sigma', type=float, default=0.1)
    parser.add_argument('--ising_bias', type=float, default=0.2)

    parser.add_argument('--eval_every', type=int, default=1)
    args = parser.parse_args()

    config = SimpleNamespace(
        vocab_size=2,
        device='cuda',
        epochs=2000,
        T=64,
        N=256,
        lr=1e-3,
        batch_size=128,
        steps_per_epoch=100,
    )

    args = SimpleNamespace(**vars(config), **vars(args))
    main(args)


