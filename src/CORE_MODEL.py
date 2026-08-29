"""
core_model.py

Defines PlasmaMCAT: a two-stream sequence model that fuses raw sensor
channels with a physics-derived domain feature (f_G) using a gated
concatenation layer, then pools over time and classifies.
"""

import torch
import torch.nn as nn


def build_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build a boolean padding mask for nn.MultiheadAttention.

    Args:
        lengths: [Batch] -- real (unpadded) length of each sequence in the batch.
        max_len: the padded sequence length (T).

    Returns:
        mask: [Batch, max_len] bool tensor, True at PADDED positions (this is
        the convention nn.MultiheadAttention's key_padding_mask expects --
        True means "ignore this position").
    """
    batch_size = lengths.shape[0]
    device = lengths.device
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, T]
    mask = positions >= lengths.unsqueeze(1)  # True where position is padding
    return mask


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the time dimension, ignoring padded positions.

    Args:
        x: [Batch, Time, Embed_Dim]
        lengths: [Batch] real length of each sequence.
    Returns:
        [Batch, Embed_Dim]
    """
    batch_size, max_len, _ = x.shape
    device = x.device
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, T]
    valid_mask = (positions < lengths.unsqueeze(1)).unsqueeze(-1).float()  # [B, T, 1]

    summed = (x * valid_mask).sum(dim=1)  # [B, Embed_Dim]
    counts = valid_mask.sum(dim=1).clamp(min=1.0)  # [B, 1], avoid div-by-zero
    return summed / counts


class GatedConcatFusion(nn.Module):
    """Gated concatenation fusion of a sensor stream and a domain stream.

    g(t) = sigmoid(Wg . h_domain(t) + bg) in (0,1)  -- a SCALAR gate per
        timestep (not a vector broadcast across embed_dim), computed from
        the domain stream only.
    h_fused(t) = W_f . [h_sensor(t) ; g(t) * h_domain(t)] + b_f
        -- concatenate the sensor embedding with the gated domain
        embedding, then project back down to embed_dim.

    Replaces an earlier multi-head cross-attention fusion layer: with a
    single physics-derived channel in the domain stream, cross-attention's
    query/key/value machinery reduces to a learned, content-dependent
    reweighting of one vector, which this module does directly with
    roughly a third of the parameters.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_sensor, x_domain, padding_mask=None):
        """
        x_sensor: [Batch, Time, Embed_Dim]
        x_domain: [Batch, Time, Embed_Dim]
        padding_mask: [Batch, Time] bool, True at padded positions. Kept in
            the signature for interface parity with the old fusion module --
            concatenation is per-timestep (no cross-timestep mixing), so
            padded positions are already zeroed upstream and are excluded
            later by masked_mean / the forecasting loss mask. No masking is
            needed inside this module.
        """
        gate = self.gate_layer(x_domain)              # [Batch, Time, 1], scalar per timestep
        gated_domain = gate * x_domain                # [Batch, Time, Embed_Dim]

        concat = torch.cat([x_sensor, gated_domain], dim=-1)  # [Batch, Time, 2*Embed_Dim]
        fused_output = self.fuse_proj(concat)          # [Batch, Time, Embed_Dim]
        return self.norm(fused_output)


class PlasmaMCAT(nn.Module):
    def __init__(self, sensor_channels=6, domain_channels=1, embed_dim=64, num_heads=4):
        """
        sensor_channels: number of raw per-timestep features fed as the
            "sensor" stream (e.g. elongation, minor_radius, triangularity --
            whatever ISN'T folded into the physics feature).
        domain_channels: number of physics-derived per-timestep features
            (e.g. 1 for f_G alone; increase if you add more derived
            indicators later).
        """
        super().__init__()
        self.sensor_embed = nn.Linear(sensor_channels, embed_dim)
        self.domain_embed = nn.Sequential(
            nn.Linear(domain_channels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.fusion_layer = GatedConcatFusion(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # raw logit
        )

    def forward(self, sensor_stream, domain_physics, lengths, per_timestep=False):
        """
        sensor_stream: [Batch, Time, sensor_channels]
        domain_physics: [Batch, Time, domain_channels]  -- e.g. f_G per timestep
        lengths: [Batch] -- real (unpadded) length of each shot in the batch
        per_timestep: if True, returns one logit per VALID timestep (for the
            forecasting task -- predicting whether disruption is imminent at
            each pre-phase timestep). If False (default), pools over time and
            returns one logit per shot (original detection-style task).

        Returns:
            per_timestep=False: [Batch, 1] raw logits
            per_timestep=True:  [Batch, Time, 1] raw logits (padded positions
                                 still present in the tensor -- mask them out
                                 using `lengths` before computing loss/metrics)
        """
        batch_size, max_len, _ = sensor_stream.shape
        padding_mask = build_padding_mask(lengths, max_len)  # [Batch, Time], True = pad

        tokens_sensor = self.sensor_embed(sensor_stream)      # [B, T, E]
        tokens_domain = self.domain_embed(domain_physics)     # [B, T, E]

        fused = self.fusion_layer(tokens_sensor, tokens_domain, padding_mask=padding_mask)  # [B, T, E]

        if per_timestep:
            return self.classifier(fused)  # [B, T, 1] -- caller must mask padded steps

        pooled = masked_mean(fused, lengths)  # [B, E] -- ignores padded timesteps
        return self.classifier(pooled)


def test_model_shapes():
    """Sanity-check the model's input/output shapes with synthetic data,
    including a batch with variable-length (padded) sequences."""
    torch.manual_seed(0)

    batch_size = 8
    max_len = 142  # longest shot in the dataset
    sensor_channels = 5   # e.g. elongation, minor_radius, triangularity, toroidal_B_field, time
    domain_channels = 1   # f_G

    model = PlasmaMCAT(sensor_channels=sensor_channels, domain_channels=domain_channels)

    sensor_stream = torch.randn(batch_size, max_len, sensor_channels)
    domain_physics = torch.randn(batch_size, max_len, domain_channels)
    # simulate real shots of varying length, e.g. one as short as 6 timesteps
    lengths = torch.tensor([142, 118, 118, 6, 90, 130, 55, 142])

    # zero out the padded region past each shot's real length, as real data prep should do
    for i, L in enumerate(lengths):
        sensor_stream[i, L:] = 0.0
        domain_physics[i, L:] = 0.0

    out = model(sensor_stream, domain_physics, lengths)
    print("Output shape:", out.shape)  # expected [8, 1]
    assert out.shape == (batch_size, 1)
    print("Shape check passed.")


if __name__ == "__main__":
    test_model_shapes()