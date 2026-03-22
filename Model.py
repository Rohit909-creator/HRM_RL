import torch
import torch.nn as nn
import pytorch_lightning as pl

# measure accuracy and other metrics
from torchmetrics import Accuracy, Precision, Recall

class HRMActionModel(pl.LightningModule):

    def __init__(self, output_size, in_channels=3, sequence_length=8,
                 embed_dim=64, N=2, T=4, learning_rate=1e-3, model_name="hrm"):
        super().__init__()
        self.model_name = model_name
        self.N = N  # number of high-level cycles
        self.T = T  # low ticks per high tick
        self.learning_rate = learning_rate

        hidden = embed_dim

        # Input projection — replaces PatchEmbedding
        # expects x: (B, sequence_length, in_channels) e.g. (B, T, 3) for [pos, vel, action]
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        # Positional embedding over sequence
        self.pos_embed = nn.Embedding(sequence_length, hidden)

        # Low GRU: takes concat(zL, zH, x) as input
        # input_size = hidden * 3 because we concat zL(hidden) + zH(hidden) + x(hidden)
        self.L_net = nn.GRUCell(input_size=hidden * 2, hidden_size=hidden)

        # High GRU: takes concat(zH, zL) as input
        self.H_net = nn.GRUCell(input_size=hidden, hidden_size=hidden)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, output_size)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = self.combined_loss
        self.train_losses, self.val_losses = [], []

    def embed_input(self, x):
        # x: (B, seq_len, in_channels)
        token_embs = self.input_proj(x)                                         # (B, seq_len, hidden)
        pos = torch.arange(x.shape[1], device=x.device)
        token_embs = token_embs + self.pos_embed(pos)                           # (B, seq_len, hidden)
        return token_embs.mean(dim=1)                                           # (B, hidden) — aggregate over sequence

    def forward(self, x, zH=None, zL=None):
        B = x.shape[0]
        hidden = self.L_net.hidden_size

        emb = self.embed_input(x)                                               # (B, hidden)

        if zH is None:
            zH = torch.zeros(B, hidden, device=x.device)
        if zL is None:
            zL = torch.zeros(B, hidden, device=x.device)

        total_ticks = self.N * self.T  # e.g. 2 * 4 = 8

        # --- bulk of recurrence under no_grad (paper section 3.3) ---
        with torch.no_grad():
            for _i in range(total_ticks - 1):
                # Low tick: conditioned on current zH
                zL = self.L_net(torch.cat([emb, zH], dim=-1), zL)
                # High tick every T low ticks
                if (_i + 1) % self.T == 0:
                    zH = self.H_net(zL, zH)

        # --- final 1-step with grad (only this step trains) ---
        zL = self.L_net(torch.cat([emb, zH], dim=-1), zL)
        zH = self.H_net(zL, zH)

        out = self.mlp(zH)
        return out, (zH.detach(), zL.detach())                                  # detach states for next supervision step

    def training_step(self, batch, batch_idx):
        x, y = batch

        B = x.shape[0]
        hidden = self.L_net.hidden_size
        zH = torch.zeros(B, hidden, device=x.device)
        zL = torch.zeros(B, hidden, device=x.device)

        total_loss = 0
        correct = 0

        for _ in range(self.N):
            y_hat, (zH, zL) = self(x, zH, zL)
            loss = self.loss_fn(y_hat, y)
            total_loss += loss

            preds = torch.argmax(y_hat, dim=1)
            correct += (preds == y).sum()

        acc = correct.float() / (y.size(0) * self.N)

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)

        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def combined_loss(self, y_hat, y):
        # Example of a combined loss if predict_action_only=False
        # y_hat: (B, output_size) where output_size = 3 (pos, vel, action)
        # y: (B, 3) where last dim is [pos, vel, action]
        mse_loss = nn.MSELoss()(y_hat[:, :2], y[:, :2])  # position and velocity
        ce_loss = nn.CrossEntropyLoss()(y_hat[:, 2:], y[:, 2].long())  # action
        return mse_loss + ce_loss



class AE(nn.Module):
    
    def __init__(self, input_size, embed_dim, output_size, dropout=0.2):
        super().__init__()
        
        self.pre_mlp= nn.Linear(input_size, embed_dim)
        
        # Simple Attn Block
        self.prenorm = nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim, 2)
        
        self.mlp= nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.final_layernorm = nn.LayerNorm(embed_dim)
        
        self.out_layernorm = nn.LayerNorm(embed_dim)
        
        self.linear_map = nn.Linear(embed_dim, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, X):
        

        x = self.relu(self.pre_mlp(X))

        x = self.prenorm(x)
        # Attention with residual connection
        attn_out, _ = self.attn(x, x, x)
        attn_out = x + attn_out  # Residual connection
        attn_out = self.out_layernorm(attn_out)  # Layer normalization
        
        # MLP with residual connection
        mlp_out = self.mlp(attn_out)
        out = attn_out + mlp_out  # Residual connection
        out = self.final_layernorm(out)  # Final layer normalization
        out = self.relu(self.linear_map(out))
        
        return out

if __name__ == "__main__":


    model = HRMActionModel(output_size=3, in_channels=3, sequence_length=8,
                 embed_dim=64, N=2, T=4, learning_rate=1e-3, model_name="hrm")
    
    x = torch.randn((1, 8, 3))
    
    out = model(x)
    
    print(out)
    
    # exit(0)
    
    # model = AE(3, 64, 3)
    
    # x = torch.randn((1,3))

    # out = model(x)
    
    # print(out)
    import os
    import pickle

    filename = "./mountaincar_demos/mountaincar_demos_20250514_022224.pkl"
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(obj[0])

    for i, actions in enumerate(obj):
        print(f"Epoch{i}: {len(actions)}")