import torch
import torch.nn as nn

# --------------------- Helper blocks (define FIRST) --------------------- #

class VariableSelection(nn.Module):
    """Selects & weights a set of variables given a context vector."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Per-time-step weights over d_in features
        self.weight_net = nn.Sequential(
            nn.Linear(d_in + d_out, d_in),
            nn.Softmax(dim=-1),
        )
        # Map weighted features → hidden
        self.linear = nn.Linear(d_in, d_out)
        # Project context to hidden
        self.ctx_proj = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, T, d_in)
        ctx: (B, 1, d_out)
        returns: (B, T, d_out)
        """
        B, T, _ = x.shape
        ctx_rep = ctx.expand(B, T, -1)                        # (B, T, d_out)
        w = self.weight_net(torch.cat([x, ctx_rep], dim=-1))  # (B, T, d_in)
        xw = self.linear(w * x)                               # (B, T, d_out)
        return xw + self.ctx_proj(ctx_rep)                    # (B, T, d_out)


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden_mult: int = 1):
        super().__init__()
        hdim = d_out * hidden_mult
        self.fc1 = nn.Linear(d_in, hdim)
        self.fc2 = nn.Linear(hdim, d_out)
        self.gate = nn.Sequential(nn.Linear(d_in, d_out), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        g = self.gate(x)
        return self.norm(residual + g * h)


class TFTLite(nn.Module):
    """
    Minimal but *correct* TFT for 1-step stock price regression.
    - Input dict with three tensors:
        past      : (B, T_past, D_past)      # OHLCV + indicators
        future    : (B, T_fut,  D_fut)       # calendar, macro (known in advance)
        static    : (B, D_static)            # ticker embedding, sector, etc.
    - Output: (B,)  → next close (or log-return)
    """
    def __init__(
        self,
        d_past: int,
        d_fut:  int,
        d_static: int,
        hidden: int = 64,
        n_heads: int = 4,
        enc_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden

        # ---------- 1. Embeddings ----------
        self.past_emb   = nn.Linear(d_past,   hidden)
        self.fut_emb    = nn.Linear(d_fut,    hidden)
        self.static_emb = nn.Linear(d_static, hidden)

        # ---------- 2. Variable Selection (VSN) ----------
        self.vsn_past  = VariableSelection(hidden, hidden)
        self.vsn_fut   = VariableSelection(hidden, hidden)
        self.vsn_static= VariableSelection(hidden, hidden)

        # ---------- 3. Temporal encoders ----------
        self.lstm_past = nn.LSTM(hidden, hidden, enc_layers,
                                 batch_first=True, dropout=dropout)

        # ---------- 4. Static context gating ----------
        self.static_gate = GatedResidualNetwork(hidden, hidden)

        # ---------- 5. Transformer decoder (causal) ----------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden*4, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # ---------- 6. Output head ----------
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden//2, 1)
        )

    def forward(self, past, future, static):
        B, T_past, _ = past.shape
        T_fut = future.shape[1]

        # ---- embed each group ----
        p = self.past_emb(past)                # (B, T_past, H)
        f = self.fut_emb(future)               # (B, T_fut,  H)
        s = self.static_emb(static)            # (B, H)

        # ---- VSN (feature weighting) ----
        p = self.vsn_past(p, s.unsqueeze(1))   # static context injected
        f = self.vsn_fut(f,  s.unsqueeze(1))

        # ---- LSTM on past only (preserves causality) ----
        memory, _ = self.lstm_past(p)          # (B, T_past, H)

        # ---- static context vector (broadcast to every step) ----
        s_ctx = self.static_gate(s)            # (B, H)
        memory = memory + s_ctx.unsqueeze(1)   # add to *every* past step

        # ---- Decoder: future-known as target sequence, causal mask ----
        tgt = f                                 # (B, T_fut, H)
        # causal mask → only attend to past + previous future-known steps
        mask = nn.Transformer.generate_square_subsequent_mask(T_fut).to(tgt.device)
        dec_out = self.decoder(tgt, memory, tgt_mask=mask)  # (B, T_fut, H)

        # ---- Predict *last* future step (1-step ahead) ----
        out = self.head(dec_out[:, -1, :])      # (B, 1)
        return out.squeeze(-1)                  # (B,)
    
