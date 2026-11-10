import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pscan import pscan


@dataclass
class MambaConfig:
    d_model: int
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False

    mup: bool = False
    mup_base_width: float = 128

    pscan: bool = True
    use_cuda: bool = False

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class LowRankCrossAlign(nn.Module):
    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(dim, rank, bias=False)
        self.k_proj = nn.Linear(dim, rank, bias=False)
        self.scale = rank ** -0.5

    def forward(self, target_seq, source_seq, source_mask=None):
        q = self.q_proj(target_seq)
        k = self.k_proj(source_seq)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if source_mask is not None:
            scores = scores.masked_fill(source_mask[:, None, :] == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, source_seq)


class AdaptiveKGFusion(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.align_k2t = LowRankCrossAlign(d_model, rank=64)
        self.eta_gate = nn.Linear(d_model * 2, d_model)
        self.theta_gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.vis_summary_gate = nn.Parameter(torch.tensor(0.1))

        nn.init.zeros_(self.eta_gate.weight)
        nn.init.constant_(self.eta_gate.bias, -2.1972)
        nn.init.zeros_(self.theta_gate.weight)
        nn.init.constant_(self.theta_gate.bias, -2.1972)

    def forward(self, t_z2t, t_k2t, k_t2k, z_t2z=None):
        if k_t2k.size(1) != t_z2t.size(1):
            k_aligned = self.align_k2t(t_z2t, k_t2k)
        else:
            k_aligned = k_t2k

        eta = torch.sigmoid(self.eta_gate(torch.cat([t_z2t, t_k2t], dim=-1)))
        theta = torch.sigmoid(self.theta_gate(torch.cat([t_z2t, k_aligned], dim=-1)))
        fused_text = self.norm(t_z2t + eta * t_k2t + theta * k_aligned)

        if z_t2z is not None:
            z_global = z_t2z.mean(dim=1, keepdim=True)
            z_t2z = z_t2z + torch.tanh(self.vis_summary_gate) * z_global
            return fused_text, z_t2z
        return fused_text


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x_v, x_q, mask_v=None, mask_q=None):
        for layer in self.layers:
            x_v, x_q = layer(x_v, x_q, mask_v=mask_v, mask_q=mask_q)
        return x_v, x_q

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.layernorm_v = LayerNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.layernorm_q = LayerNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward_unimodal(self, x):
        output = self.mixer.forward_unimodal(self.norm.forward_unimodal(x))
        return output + x

    def forward(self, x_v, x_q, mask_v=None, mask_q=None):
        output_v = self.layernorm_v(x_v)
        output_q = self.layernorm_q(x_q)
        output_v, output_q = self.mixer(output_v, output_q, mask_v=mask_v, mask_q=mask_q)
        return output_v + x_v, output_q + x_q

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm.forward_unimodal(x), cache)
        return output + x, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.in_proj_q = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.dropout = nn.Dropout(p=0.1)

        self.align_q2v = LowRankCrossAlign(config.d_model, rank=64)
        self.align_v2q = LowRankCrossAlign(config.d_model, rank=64)
        self.mod_v = nn.Linear(config.d_model, config.d_model * 2)
        self.mod_q = nn.Linear(config.d_model, config.d_model * 2)
        nn.init.zeros_(self.mod_v.weight)
        nn.init.zeros_(self.mod_v.bias)
        nn.init.zeros_(self.mod_q.weight)
        nn.init.zeros_(self.mod_q.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.conv1d_q = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        self.x_proj_q = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj_q = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        for dt_p in (self.dt_proj, self.dt_proj_q):
            if config.dt_init == "constant":
                nn.init.constant_(dt_p.weight, dt_init_std)
            elif config.dt_init == "random":
                nn.init.uniform_(dt_p.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            dt = torch.exp(
                torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
            ).clamp(min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_p.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        self.A_log_q = nn.Parameter(torch.log(A.clone()))
        self.A_log_q._no_weight_decay = True
        self.D_q = nn.Parameter(torch.ones(config.d_inner))
        self.D_q._no_weight_decay = True

        self.bi_gate_v = nn.Parameter(torch.zeros(1, 1, config.d_inner))
        self.bi_gate_q = nn.Parameter(torch.zeros(1, 1, config.d_inner))

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.out_proj_q = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm.forward_unimodal(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm.forward_unimodal(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm.forward_unimodal(C)
        return dt, B, C

    def _bidirectional_scan(self, x_in, in_proj, conv1d, x_proj, dt_proj, A_log, D, bi_gate):
        _, L, _ = x_in.shape
        xz = in_proj(x_in)
        x, z = xz.chunk(2, dim=-1)

        x_fwd = F.silu(conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2))
        y_fwd = self._ssm_branch(x_fwd, z, x_proj, dt_proj, A_log, D)

        x_rev = torch.flip(x, dims=[1])
        x_bwd = F.silu(conv1d(x_rev.transpose(1, 2))[:, :, :L].transpose(1, 2))
        y_bwd = torch.flip(
            self._ssm_branch(x_bwd, torch.flip(z, dims=[1]), x_proj, dt_proj, A_log, D),
            dims=[1],
        )

        w = torch.sigmoid(bi_gate)
        y = w * y_fwd + (1.0 - w) * y_bwd
        return y, z

    def forward_unimodal(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = F.silu(x.transpose(1, 2))
        y = self.ssm(x, z)
        if self.config.use_cuda:
            return self.out_proj(y)
        return self.out_proj(y * F.silu(z))

    def forward(self, x_v, x_q, mask_v=None, mask_q=None):
        _, L_v, _ = x_v.shape
        _, L_q, _ = x_q.shape

        q_for_v = self.align_q2v(x_v, x_q, source_mask=mask_q)
        v_for_q = self.align_v2q(x_q, x_v, source_mask=mask_v)

        gamma_v, beta_v = self.mod_v(q_for_v).chunk(2, dim=-1)
        x_v_mod = x_v * (1.0 + torch.tanh(gamma_v)) + beta_v

        gamma_q, beta_q = self.mod_q(v_for_q).chunk(2, dim=-1)
        x_q_mod = x_q * (1.0 + torch.tanh(gamma_q)) + beta_q

        y_v, z_v = self._bidirectional_scan(
            x_v_mod, self.in_proj, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D, self.bi_gate_v
        )
        y_q, z_q = self._bidirectional_scan(
            x_q_mod, self.in_proj_q, self.conv1d_q, self.x_proj_q, self.dt_proj_q, self.A_log_q, self.D_q, self.bi_gate_q
        )

        z_q_aligned = self.downsample(z_q, L_v) if L_q != L_v else z_q
        z_v_aligned = self.downsample(z_v, L_q) if L_v != L_q else z_v

        qv_v = F.silu(z_v) * torch.sigmoid(z_q_aligned)
        qv_q = F.silu(z_q) * torch.sigmoid(z_v_aligned)

        out_v = self.out_proj(self.dropout(y_v * qv_v))
        out_q = self.out_proj_q(self.dropout(y_q * qv_q))
        return out_v, out_q

    def downsample(self, x, output_size):
        return F.adaptive_avg_pool1d(x.permute(0, 2, 1), output_size=output_size).permute(0, 2, 1)

    def upsample(self, x, output_size):
        return F.interpolate(x.permute(0, 2, 1), size=output_size, mode='linear', align_corners=True).permute(0, 2, 1)

    def _ssm_branch(self, x, z, x_proj, dt_proj, A_log, D_param):
        orig_dtype = x.dtype
        A = -torch.exp(A_log.float())
        D = D_param.float()

        deltaBC = x_proj(x)
        delta, B, C = torch.split(
            deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1
        )
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = dt_proj.weight @ delta.transpose(1, 2)

        if self.config.use_cuda:
            x_c = x.transpose(1, 2)
            B_c = B.transpose(1, 2)
            C_c = C.transpose(1, 2)
            z_c = z.transpose(1, 2)
            y = self.selective_scan_cuda(
                x_c, delta, A, B_c, C_c, D, z=z_c, delta_softplus=True, delta_bias=dt_proj.bias.float()
            )
            return y.transpose(1, 2).to(orig_dtype)

        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + dt_proj.bias)

        if self.config.pscan:
            y = self.selective_scan(x.float(), delta.float(), A, B.float(), C.float(), D)
        else:
            y = self.selective_scan_seq(x.float(), delta.float(), A, B.float(), C.float(), D)
        return y.to(orig_dtype)

    def ssm(self, x, z):
        return self._ssm_branch(x, z, self.x_proj, self.dt_proj, self.A_log, self.D)

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * (x.unsqueeze(-1))
        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * (x.unsqueeze(-1))
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device, dtype=deltaA.dtype)
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x

    def step(self, x, cache):
        h, inputs = cache
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]
        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        z = F.silu(z)
        output = self.out_proj(y * z)
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        return output, (h, inputs)

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(
            deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1
        )
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)
        BX = deltaB * (x.unsqueeze(-1))
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        h = deltaA * h + BX
        y = (h @ C.unsqueeze(-1)).squeeze(2)
        return (y + D * x).to(x.dtype), h


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()
        self.use_mup = use_mup
        self.eps = eps
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward_unimodal(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight if not self.use_mup else output

    def forward(self, x_v, x_q):
        output_v = x_v * torch.rsqrt(x_v.pow(2).mean(-1, keepdim=True) + self.eps)
        output_q = x_q * torch.rsqrt(x_q.pow(2).mean(-1, keepdim=True) + self.eps)
        if not self.use_mup:
            return output_v * self.weight, output_q * self.weight
        return output_v, output_q


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()
        self.eps = eps
        self.use_mup = use_mup
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        if not self.use_mup:
            return norm_x * self.weight + self.bias
        return norm_x