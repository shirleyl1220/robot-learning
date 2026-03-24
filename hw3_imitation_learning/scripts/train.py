"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
# from hw3_imitation_learning.hw3 import model
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# TODO: Choose your own hyperparameters!
EPOCHS = 500
BATCH_SIZE = 128  # try powers of 2 like 64, 128, 256, etc.
LR = 1e-4
VAL_SPLIT = 0.1


def debug_data(states: np.ndarray, actions: np.ndarray, normalizer, state_keys, action_keys) -> None:
    print("\n=== DATA DEBUG ===")

    # 1. Check for NaNs/Infs in raw data
    print(f"States NaN: {np.isnan(states).any()}, Inf: {np.isinf(states).any()}")
    print(f"Actions NaN: {np.isnan(actions).any()}, Inf: {np.isinf(actions).any()}")

    # 2. Flag near-zero-variance state dims (will be ~noise after normalization)
    low_var = np.where(states.std(axis=0) < 1e-4)[0]
    if len(low_var):
        print(f"WARNING: near-constant state dims (std < 1e-4): {low_var.tolist()}")
    else:
        print("State dims: all have reasonable variance")

    # 3. Action statistics in raw space
    print(f"Action mean: {actions.mean(axis=0).round(4)}")
    print(f"Action std:  {actions.std(axis=0).round(4)}")
    print(f"Action min:  {actions.min(axis=0).round(4)}")
    print(f"Action max:  {actions.max(axis=0).round(4)}")

    # 4. Normalized action range sanity check (should be roughly [-3, 3])
    norm_actions = normalizer.normalize_action(actions)
    print(f"Normalized action std:  {norm_actions.std(axis=0).round(3)}")
    print(f"Normalized action max abs: {np.abs(norm_actions).max(axis=0).round(3)}")

    # 5. Goal distribution (if state_goal is in state_keys)
    if state_keys and any("state_goal" in k for k in state_keys):
        goal_idx = sum(
            (3 if "[:3]" in k or "original_pos" in k or "state_ee_xyz" in k else
             1 if "state_gripper" in k else 3)
            for k in state_keys
            if "state_goal" not in k and k != list(filter(lambda x: "state_goal" in x, state_keys))[0]
        )
        # Simpler: just print raw state_goal column counts
        goal_cols = states[:, 9:12] if states.shape[1] > 11 else None
        if goal_cols is not None:
            counts = goal_cols.sum(axis=0)
            print(f"Goal class counts (red/green/blue) at idx 9:12: {counts.astype(int)}")

    print("==================\n")


@torch.no_grad()
def debug_predictions(model, loader, device, epoch) -> None:
    """Print prediction vs target stats for first batch."""
    model.eval()
    states, action_chunks = next(iter(loader))
    states, action_chunks = states.to(device), action_chunks.to(device)
    pred = model(states)

    pred_mean = pred.mean().item()
    pred_std = pred.std().item()
    target_std = action_chunks.std().item()
    per_dim_mse = ((pred - action_chunks) ** 2).mean(dim=(0, 1))

    print(f"\n[Epoch {epoch}] Prediction debug:")
    print(f"  pred  mean={pred_mean:.4f}  std={pred_std:.4f}")
    print(f"  target std={target_std:.4f}  (should be ~1 if normalized)")
    print(f"  per action-dim MSE: {per_dim_mse.cpu().numpy().round(4)}")

    grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  grad norm (last step): {grad_norm:.4f}")


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states, action_chunks = states.to(device), action_chunks.to(device)
        #compute loss
        loss = model.compute_loss(states, action_chunks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states, action_chunks = states.to(device), action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1  


    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--extra-zarr",
        nargs="+",
        default=None,
        help="Additional zarr stores to merge.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")
    debug_data(states, actions, normalizer, args.state_keys, args.action_keys)

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # TODO: implement an optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        if epoch == 1 or epoch % 50 == 0:
            debug_predictions(model, val_loader, device, epoch)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
