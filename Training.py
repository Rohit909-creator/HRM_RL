from mountaincar_dataset import get_dataloaders
import pytorch_lightning as pl
from Model import HRMActionModel

train_loader, val_loader, test_loader, info = get_dataloaders(
    "./mountaincar_demos/mountaincar_demos_20250514_022224.pkl",
    sequence_length=8,
    predict_action_only=True,   # y = next action int → CrossEntropyLoss(output_size=3)
    batch_size=64,
)

model = HRMActionModel(
    output_size=info["n_actions"],   # 3
    in_channels=3,                   # [pos, vel, action]
    sequence_length=info["sequence_length"],
    embed_dim=64,
    N=2,
    T=4,
)

trainer = pl.Trainer(
    max_epochs=20,
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    deterministic=False,
)

trainer.fit(model, train_loader, val_loader)