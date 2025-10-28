# models/vision_bdh_v2_cls.py
# MIT License – same as the repo

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bdh import BDHConfig, get_freqs


class BidirectionalAttentionV2(nn.Module):
    """
    Bidirectional Attention (v2) for Vision-BDH.
    Identical to the one used in vision_bdh_v2.py (Q = K constraint),
    with optional softmax for stability.
    """
    def __init__(self, config: BDHConfig, use_softmax: bool = True):
        super().__init__()
        self.config = config
        self.use_softmax = use_softmax

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        # Rotary frequencies (non-trainable)
        self.freqs = nn.Parameter(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N),
            requires_grad=False,
        )

    @staticmethod
    def phases_cos_sin(phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phases = (phases % 1) * (2 * torch.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Rotate last dim pairs: (..., 2k) <-> (..., 2k+1)
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos, sin = BidirectionalAttentionV2.phases_cos_sin(phases)
        return (v * cos) + (v_rot * sin)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Shapes:
        #   Q, K: (B, nh, T, N) with Q is K (Q=K constraint)
        #   V:    (B, nh_or_1, T, D)  (broadcast along heads)
        assert K is Q, "In Vision-BDH, K must equal Q"
        _, _, T, _ = Q.size()

        # Rotary phases: (1, 1, T, N)
        r_phases = (torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
                    .view(1, 1, -1, 1)) * self.freqs

        QR = self.rope(r_phases, Q)
        KR = QR  # Q = K

        scores = QR @ KR.mT  # (B, nh, T, T)

        if self.use_softmax:
            scores = F.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)

        return scores @ V  # (B, nh, T, D)


class VisionBDHv2CLS(nn.Module):
    """
    Vision-BDH v2 with a learnable [CLS] token and a CLS-based classification head.

    Differences vs. the repo's VisionBDHv2:
      - Add `cls_token` and extend `pos_embed` to length (1 + num_patches)
      - Prepend CLS to token sequence before BDH core
      - Replace mean pooling with selecting CLS representation
    """
    def __init__(
        self,
        bdh_config: BDHConfig,
        img_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        in_channels: int = 3,
        use_softmax_attn: bool = True,
    ):
        super().__init__()
        self.config = bdh_config
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.use_softmax_attn = use_softmax_attn

        D = bdh_config.n_embd

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, D, kernel_size=patch_size, stride=patch_size
        )

        # --- NEW: CLS token + position embedding includes CLS position ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, D) * 0.02
        )

        # BDH core (recurrent)
        self.build_bdh_layers()

        # Classification head (same head dims, but fed with CLS instead of mean)
        self.ln_final = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.head = nn.Linear(D, num_classes)

        # Init
        self.apply(self._init_weights)
        nn.init.zeros_(self.cls_token)  # common ViT practice

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def build_bdh_layers(self) -> None:
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = C.mlp_internal_dim_multiplier * D // nh

        # Decoder/encoder weights
        self.decoder = nn.Parameter(torch.empty((nh * N, D)))
        nn.init.xavier_uniform_(self.decoder)

        self.encoder = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder)

        self.encoder_v = nn.Parameter(torch.empty((nh, D, N)))
        nn.init.xavier_uniform_(self.encoder_v)

        # Attention and norms
        self.attn = BidirectionalAttentionV2(C, use_softmax=self.use_softmax_attn)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(C.dropout)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        Cfg = self.config
        D = Cfg.n_embd
        nh = Cfg.n_head
        N = D * Cfg.mlp_internal_dim_multiplier // nh

        # 1) Patchify -> tokens (B, T, D)
        x = self.patch_embed(x)                      # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)             # (B, T, D); T = num_patches

        # 2) Prepend CLS and add pos-emb
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x = torch.cat([cls, x], dim=1)               # (B, 1 + T, D)
        x = x + self.pos_embed                       # absolute pos embedding

        # 3) BDH recurrent core expects (B, 1, T, D)
        x = x.unsqueeze(1)                           # (B, 1, 1 + T, D)

        for _ in range(Cfg.n_layer):
            x = self.ln(x)                      # (B, 1, 1 + T, D)

            # Project to per-head latent
            x_latent = x @ self.encoder         # (B, nh, 1 + T, N)
            x_sparse = F.relu(x_latent)

            # Bidirectional attention (Q = K)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)  # (B, nh, 1+T, D)
            yKV = self.ln(yKV)

            # Value-side projection and sparsity
            y_latent = yKV @ self.encoder_v          # (B, nh, 1 + T, N)
            y_sparse = F.relu(y_latent)

            # Gated-like interaction
            xy_sparse = x_sparse * y_sparse          # (B, nh, 1 + T, N)
            xy_sparse = self.drop(xy_sparse)

            # Decoder aggregates heads back to D
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, -1, N * nh) @ self.decoder  # (B, 1, 1+T, D)
            y = self.ln(yMLP)

            # Residual (post-norm)
            x = self.ln(x + y)

        # 4) Classification on CLS token (index 0 along tokens)
        x = x.squeeze(1)                 # (B, 1 + T, D)
        cls_rep = x[:, 0, :]             # (B, D)
        cls_rep = self.ln_final(cls_rep)
        logits = self.head(cls_rep)      # (B, num_classes)

        return logits
