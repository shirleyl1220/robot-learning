import argparse
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick trajectory inspection for single-cube or multicube zarr datasets.")
    parser.add_argument(
        "--zarr",
        type=Path,

        default=Path("/Users/shirley/Documents/SCHOOL/SPRING26/robotlearning/robot-learning/hw3_imitation_learning/datasets/processed/multi_cube/processed_ee_xyz.zarr"),
        help="Path to a zarr store (processed or raw teleop).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=60,
        help="Maximum number of episode trajectories to plot (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z = zarr.open_group(str(args.zarr), mode="r")

    print(f"Dataset: {args.zarr}")
    print("Groups:", list(z.group_keys()))
    print("data arrays:", list(z["data"].array_keys()))
    print("meta arrays:", list(z["meta"].array_keys()))

    goal = np.array(z['data/state_goal'])
    print("Red episodes:", (goal[:,0]==1).sum())
    print("Green episodes:", (goal[:,1]==1).sum())  
    print("Blue episodes:", (goal[:,2]==1).sum())
    data_keys = set(z["data"].array_keys())

    red = np.array(z['data/original_pos_cube_red'])
    print("red cube x range:", red[:,0].min(), red[:,0].max())
    print("red cube y range:", red[:,1].min(), red[:,1].max())
    goal = np.array(z['data/goal_pos'])
    print("bin x range:", goal[:,0].min(), goal[:,0].max())
    print("bin y range:", goal[:,1].min(), goal[:,1].max())
    print ("goal labels (first 10):")
    goal = np.array(z['data/state_goal'])[:5]
    print(goal)

    import torch
    ckpt = torch.load("/Users/shirley/Documents/SCHOOL/SPRING26/robotlearning/robot-learning/hw3_imitation_learning/checkpoints/multi_cube/best_model_ee_xyz_multitask.pt", map_location="cpu",weights_only=False)
    print(ckpt['val_loss'])
    print(ckpt['state_keys'])
    print(ckpt['epoch'])

    if "state_ee_xyz" in data_keys:
        ee = np.array(z["data/state_ee_xyz"])
        ee_name = "state_ee_xyz"
    elif "state_ee" in data_keys:
        ee = np.array(z["data/state_ee"])[:, :3]
        ee_name = "state_ee[:3]"
    else:
        raise KeyError("Could not find state_ee_xyz or state_ee in data group.")

    if "action_ee_xyz" in data_keys:
        print("action_ee_xyz mean:", np.array(z["data/action_ee_xyz"]).mean(axis=0))
    if "state_cube" in data_keys:
        print("state_cube std:", np.array(z["data/state_cube"]).std(axis=0))

    ends = np.array(z["meta/episode_ends"])
    lengths = np.diff(np.concatenate([[0], ends]))
    print(f"episodes: {len(ends)}")
    print("mean episode length:", float(lengths.mean()))
    print("min:", int(lengths.min()), "max:", int(lengths.max()))
    print("using ee key:", ee_name)

    goal = np.array(z['data/state_goal'])

    starts = np.concatenate([[0], ends[:-1]])
    ends = np.array(z["meta/episode_ends"])

    


    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping trajectory plot.")
        return

    starts = np.concatenate([[0], ends[:-1]])
    n_plot = min(args.max_episodes, len(ends))

    # If multicube goal labels exist, color by goal class.
    has_goal = "state_goal" in data_keys
    goal_names = ["red", "green", "blue"]
    goal_colors = ["tab:red", "tab:green", "tab:blue"]

    # Optional multicube diagnostic:
    # compare EE position at episode midpoint vs selected goal-cube position.
    if has_goal:
        if "original_pos_cube_red" in data_keys:
            cube_red = np.array(z["data/original_pos_cube_red"])[:, :3]
            cube_green = np.array(z["data/original_pos_cube_green"])[:, :3]
            cube_blue = np.array(z["data/original_pos_cube_blue"])[:, :3]
        elif "pos_cube_red" in data_keys:
            cube_red = np.array(z["data/pos_cube_red"])[:, :3]
            cube_green = np.array(z["data/pos_cube_green"])[:, :3]
            cube_blue = np.array(z["data/pos_cube_blue"])[:, :3]
        else:
            cube_red = cube_green = cube_blue = None

        if cube_red is not None:
            goal_onehot = np.array(z["data/state_goal"])
            midpoint_dists = []
            for i, (s, e) in enumerate(zip(starts, ends)):
                mid = (int(s) + int(e) - 1) // 2
                goal_idx = int(np.argmax(goal_onehot[s]))
                if goal_idx == 0:
                    goal_pos = cube_red[mid]
                elif goal_idx == 1:
                    goal_pos = cube_green[mid]
                else:
                    goal_pos = cube_blue[mid]

                ee_mid = ee[mid, :3]
                dist = float(np.linalg.norm(ee_mid - goal_pos))
                midpoint_dists.append(dist)
                if i < 10:
                    print(
                        f"ep {i:02d} goal={goal_names[goal_idx]} | "
                        f"||ee_mid - goal_cube_mid|| = {dist:.4f}"
                    )

            midpoint_dists = np.array(midpoint_dists, dtype=np.float32)
            print(
                "midpoint distance summary (EE vs goal cube) "
                f"mean={midpoint_dists.mean():.4f}, "
                f"p50={np.median(midpoint_dists):.4f}, "
                f"p90={np.quantile(midpoint_dists, 0.9):.4f}"
            )

    plt.figure(figsize=(14, 7))
    for i, (s, e) in enumerate(zip(starts[:n_plot], ends[:n_plot])):
        if has_goal:
            g = np.array(z["data/state_goal"])[s]
            goal_idx = int(np.argmax(g))
            color = goal_colors[goal_idx]
            label = f"ep {i} ({goal_names[goal_idx]})"
        else:
            color = None
            label = f"ep {i}"

        plt.plot(
            ee[s:e, 0],
            ee[s:e, 1],
            alpha=0.7,
            color=color,
            label=label,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"EE trajectories top view ({n_plot} episodes)")
    if n_plot <= 20:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()