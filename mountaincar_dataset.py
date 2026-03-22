import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader, random_split


class MountainCarDataset(Dataset):
    """
    Sliding window dataset over mountain car demo episodes.

    Each sample:
        x : (sequence_length, 3)  — window of [pos, vel, action] steps
        y : (3,)                  — next step [pos, vel, action] as regression target
                                    OR just next action (int) for classification

    Args:
        episodes       : raw pickle data — list of episodes,
                         each episode = list of (np.array([pos, vel]), action) tuples
        sequence_length: number of past steps fed as context (window size)
        predict_action_only: if True, y = next action int (for CrossEntropyLoss)
                             if False, y = [pos, vel, action] float (for MSE + CE combo)
    """

    def __init__(self, episodes, sequence_length=8, predict_action_only=False):
        self.sequence_length = sequence_length
        self.predict_action_only = predict_action_only
        self.samples = []  # list of (x_window, y_target)

        self._build_samples(episodes)

    def _build_samples(self, episodes):
        for episode in episodes:
            if len(episode) < self.sequence_length + 1:
                # episode too short to form even one window — skip
                continue

            # flatten each timestep into [pos, vel, action]
            flat = []
            for state, action in episode:
                pos, vel = float(state[0]), float(state[1])
                flat.append([pos, vel, float(action)])

            flat = np.array(flat, dtype=np.float32)  # (T, 3)

            # sliding window: window [i : i+seq_len] → predict step [i+seq_len]
            for i in range(len(flat) - self.sequence_length):
                x_window = flat[i : i + self.sequence_length]          # (seq_len, 3)
                next_step = flat[i + self.sequence_length]              # (3,)

                if self.predict_action_only:
                    y = int(next_step[2])                               # scalar int
                else:
                    y = next_step                                       # (3,) float

                self.samples.append((x_window, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)                       # (seq_len, 3)

        if self.predict_action_only:
            y = torch.tensor(y, dtype=torch.long)                      # scalar — for CrossEntropyLoss
        else:
            y = torch.tensor(y, dtype=torch.float32)                   # (3,) — for MSE/combined loss

        return x, y


def get_dataloaders(
    pickle_path,
    sequence_length=8,
    predict_action_only=True,
    batch_size=64,
    val_split=0.15,
    test_split=0.05,
    num_workers=0,
    seed=42,
):
    """
    Loads the pickle file, builds the dataset, splits into train/val/test,
    and returns three DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, dataset_info (dict)
    """
    assert os.path.exists(pickle_path), f"File not found: {pickle_path}"

    with open(pickle_path, "rb") as f:
        episodes = pickle.load(f)

    print(f"Loaded {len(episodes)} episodes")
    for i, ep in enumerate(episodes[:3]):
        print(f"  Episode {i}: {len(ep)} steps")

    dataset = MountainCarDataset(
        episodes,
        sequence_length=sequence_length,
        predict_action_only=predict_action_only,
    )

    total = len(dataset)
    n_test = max(1, int(total * test_split))
    n_val  = max(1, int(total * val_split))
    n_train = total - n_val - n_test

    print(f"\nDataset split — total: {total} | train: {n_train} | val: {n_val} | test: {n_test}")

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # useful stats to pass to your model / logging
    dataset_info = {
        "total_samples"  : total,
        "n_train"        : n_train,
        "n_val"          : n_val,
        "n_test"         : n_test,
        "sequence_length": sequence_length,
        "n_episodes"     : len(episodes),
        "input_shape"    : (sequence_length, 3),
        # action space: 0=push left, 1=no push, 2=push right
        "n_actions"      : 3,
    }

    return train_loader, val_loader, test_loader, dataset_info


# ------------------------------------------------------------------
# Sanity check — run directly: python mountaincar_dataset.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    pickle_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "./mountaincar_demos/mountaincar_demos_20250514_022224.pkl"
    )

    train_loader, val_loader, test_loader, info = get_dataloaders(
        pickle_path,
        sequence_length=8,
        predict_action_only=False,   # set False if you want to predict full next state
        batch_size=64,
    )

    print(f"\nDataset info: {info}")

    # inspect one batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  x shape : {x_batch.shape}   dtype: {x_batch.dtype}")  # (B, seq_len, 3)
    print(f"  y shape : {y_batch.shape}   dtype: {y_batch.dtype}")  # (B,) long  OR  (B, 3) float
    print(f"  x[0]    :\n{x_batch[0]}")
    print(f"  y[0]    : {y_batch[0]}")

    # confirm action distribution in one batch
    if y_batch.dtype == torch.long:
        for a in range(3):
            pct = (y_batch == a).float().mean().item() * 100
            print(f"  action {a} frequency: {pct:.1f}%")