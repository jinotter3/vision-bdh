# models/vision_bdh_v2_cls_optimized.py
# Optimized version for transformer-level training speed
# MIT License – same as the repo

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.bdh import BDHConfig, get_freqs


class BidirectionalAttentionV2Optimized(nn.Module):
    """
    Optimized Bidirectional Attention with cached RoPE and optional SDPA.
    """
    def __init__(self, config: BDHConfig, use_softmax: bool = True, max_seq_len: int = 128):
        super().__init__()
        self.config = config
        self.use_softmax = use_softmax
        self.max_seq_len = max_seq_len

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        # Pre-compute rotary embeddings for max sequence length
        freqs = get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        positions = torch.arange(0, max_seq_len, dtype=torch.float32).view(1, 1, -1, 1)
        r_phases = positions * freqs
        
        # Pre-compute cos and sin
        phases_normalized = (r_phases % 1) * (2 * torch.pi)
        cos_cached = torch.cos(phases_normalized)
        sin_cached = torch.sin(phases_normalized)
        
        # Register as buffers (non-trainable, moved to device automatically)
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    @staticmethod
    @torch.jit.script
    def rope_cached(cos: torch.Tensor, sin: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Fused RoPE using cached cos/sin values."""
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).flatten(-2)
        return (v * cos) + (v_rot * sin)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with cached RoPE and optional flash attention.
        
        Args:
            Q, K: (B, nh, T, N) with Q is K (Q=K constraint)
            V: (B, nh_or_1, T, D)
        Returns:
            (B, nh, T, D)
        """
        assert K is Q, "In Vision-BDH, K must equal Q"
        B, nh, T, N = Q.size()

        # Use pre-computed cos/sin (slice to current sequence length)
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]

        # Apply RoPE with cached values
        QR = self.rope_cached(cos, sin, Q)
        KR = QR  # Q = K constraint

        # Expand V once
        V_expanded = V.expand(-1, nh, -1, -1)

        if self.use_softmax:
            # Use PyTorch's optimized SDPA when possible
            # Reshape for SDPA: (B, nh, T, N) -> (B, nh, T, N)
            # Note: SDPA expects (B, H, T, D) format
            try:
                # Try to use flash attention if available
                output = F.scaled_dot_product_attention(
                    QR, KR, V_expanded,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0 / (N ** 0.5)
                )
                return output
            except:
                # Fallback to manual computation
                scores = QR @ KR.mT / (N ** 0.5)
                scores = F.softmax(scores, dim=-1)
                return scores @ V_expanded
        else:
            # No softmax - direct computation
            scores = QR @ KR.mT
            return scores @ V_expanded


class FusedLinearReLU(nn.Module):
    """Fused Linear + ReLU for better performance."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


class VisionBDHv2CLSOptimized(nn.Module):
    """
    Optimized Vision-BDH v2 with performance improvements:
    - Cached RoPE embeddings
    - Fused operations where possible
    - Optimized memory layout
    - Flash attention support
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
        self.max_seq_len = 1 + self.num_patches  # CLS + patches

        D = bdh_config.n_embd

        # Patch embedding - use channels_last for better performance
        self.patch_embed = nn.Conv2d(
            in_channels, D, kernel_size=patch_size, stride=patch_size
        )

        # CLS token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, D) * 0.02
        )

        # BDH core (recurrent)
        self.build_bdh_layers()

        # Classification head
        self.ln_final = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.head = nn.Linear(D, num_classes)

        # Init
        self.apply(self._init_weights)
        nn.init.zeros_(self.cls_token)

    def build_bdh_layers(self) -> None:
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = C.mlp_internal_dim_multiplier * D // nh

        # Store shapes for efficient reshaping
        self.nh = nh
        self.D = D
        self.N = N

        # Optimized weight layout - use nn.Linear for better kernel fusion
        # Instead of raw parameters, use Linear layers (can be fused better)
        self.encoder_layers = nn.ModuleList([
            nn.Linear(D, N, bias=False) for _ in range(nh)
        ])
        self.encoder_v_layers = nn.ModuleList([
            nn.Linear(D, N, bias=False) for _ in range(nh)
        ])
        
        # Decoder as single linear layer
        self.decoder = nn.Linear(nh * N, D, bias=False)

        # Attention with cached RoPE
        self.attn = BidirectionalAttentionV2Optimized(
            C, 
            use_softmax=self.use_softmax_attn,
            max_seq_len=self.max_seq_len
        )
        
        # Layer norms
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(C.dropout)

        # Initialize encoder/decoder layers
        for layer in self.encoder_layers + self.encoder_v_layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass.
        
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        nh = self.nh
        D = self.D
        N = self.N

        # 1) Patchify -> tokens (B, T, D)
        x = self.patch_embed(x)                      # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)             # (B, T, D)

        # 2) Prepend CLS and add pos-emb
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x = torch.cat([cls, x], dim=1)               # (B, 1 + T, D)
        x = x + self.pos_embed                       # absolute pos embedding

        # 3) BDH recurrent core
        seq_len = x.shape[1]  # 1 + num_patches
        x = x.unsqueeze(1)    # (B, 1, seq_len, D)

        for _ in range(self.config.n_layer):
            # Pre-norm
            x_norm = self.ln(x)  # (B, 1, seq_len, D)
            x_2d = x_norm.squeeze(1)  # (B, seq_len, D)

            # Parallel projection to all heads
            # Stack encoder outputs: list of (B, seq_len, N) -> (B, nh, seq_len, N)
            x_latent = torch.stack([
                self.encoder_layers[h](x_2d) for h in range(nh)
            ], dim=1)
            x_sparse = F.relu(x_latent)

            # Bidirectional attention (Q = K)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x_norm)  # (B, nh, seq_len, D)
            yKV = self.ln(yKV)

            # Value-side projection (parallel)
            yKV_reshaped = yKV.transpose(1, 2).reshape(B, seq_len, nh * D)
            y_latent_list = []
            for h in range(nh):
                y_h = self.encoder_v_layers[h](yKV[:, h, :, :])  # (B, seq_len, N)
                y_latent_list.append(y_h)
            y_latent = torch.stack(y_latent_list, dim=1)  # (B, nh, seq_len, N)
            y_sparse = F.relu(y_latent)

            # Gated interaction
            xy_sparse = x_sparse * y_sparse  # (B, nh, seq_len, N)
            xy_sparse = self.drop(xy_sparse)

            # Decoder: (B, nh, seq_len, N) -> (B, seq_len, nh*N) -> (B, seq_len, D)
            xy_flat = xy_sparse.permute(0, 2, 1, 3).reshape(B, seq_len, nh * N)
            yMLP = self.decoder(xy_flat)  # (B, seq_len, D)
            yMLP = yMLP.unsqueeze(1)  # (B, 1, seq_len, D)
            
            y = self.ln(yMLP)

            # Residual
            x = self.ln(x + y)

        # 4) Classification on CLS token
        x = x.squeeze(1)                 # (B, seq_len, D)
        cls_rep = x[:, 0, :]             # (B, D)
        cls_rep = self.ln_final(cls_rep)
        logits = self.head(cls_rep)      # (B, num_classes)

        return logits
