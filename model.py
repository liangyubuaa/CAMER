# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, SwinModel

from parameters import CHANNEL_ORDERS, SCL_PARAMS, build_model_params


def normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Return the normalized graph Laplacian I - D^{-1/2} A D^{-1/2}."""
    degree = torch.sum(adj, dim=1)
    degree_inv_sqrt = 1.0 / torch.sqrt(degree + 1e-5)
    degree_matrix = torch.diag_embed(degree_inv_sqrt)
    identity = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    return identity - torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)


def _adjacency_from_edges(channel_list: List[str], adjacency_list: Dict[str, List[str]]) -> torch.Tensor:
    index = {channel: i for i, channel in enumerate(channel_list)}
    adj = torch.zeros(len(channel_list), len(channel_list), dtype=torch.float32)

    for src, neighbors in adjacency_list.items():
        if src not in index:
            continue
        for dst in neighbors:
            if dst in index:
                adj[index[src], index[dst]] = 1.0
                adj[index[dst], index[src]] = 1.0

    adj.fill_diagonal_(1.0)
    return adj


def default_eeg_adjacency(num_electrodes: int, dataset_name: Optional[str] = None) -> torch.Tensor:
    """Return a topology prior for the supported EEG montages."""
    dataset_key = dataset_name.lower() if dataset_name is not None else None

    deap_mahnob_edges = {
        "FP1": ["AF3"], "FP2": ["AF4"], "AF3": ["FP1", "FZ", "F3"],
        "AF4": ["FP2", "F4", "FZ"], "F7": ["FC5"],
        "F3": ["AF3", "FC1", "FC5"], "FZ": ["AF3", "AF4", "FC2", "FC1"],
        "F4": ["AF4", "FC6", "FC2"], "F8": ["FC6"],
        "FC5": ["F7", "F3", "C3", "T7"], "FC1": ["F3", "FZ", "CZ", "C3"],
        "FC2": ["FZ", "F4", "C4", "CZ"], "FC6": ["F4", "F8", "T8", "C4"],
        "T7": ["FC5", "CP5"], "C3": ["FC5", "FC1", "CP1", "CP5"],
        "CZ": ["FC1", "FC2", "CP2", "CP1"], "C4": ["FC2", "FC6", "CP6", "CP2"],
        "T8": ["FC6", "CP6"], "CP5": ["T7", "C3", "P3", "P7"],
        "CP1": ["C3", "CZ", "PZ", "P3"], "CP2": ["CZ", "C4", "P4", "PZ"],
        "CP6": ["C4", "T8", "P8", "P4"], "P7": ["CP5"],
        "P3": ["CP5", "CP1", "PO3"], "PZ": ["CP1", "CP2", "PO4", "PO3"],
        "P4": ["CP2", "CP6", "PO4"], "P8": ["CP6"],
        "PO3": ["P3", "PZ", "OZ", "O1"], "PO4": ["PZ", "P4", "O2", "OZ"],
        "O1": ["PO3", "OZ"], "OZ": ["PO3", "PO4", "O2", "O1"],
        "O2": ["PO4", "OZ"],
    }
    amigos_edges = {
        "AF3": ["F3", "AF4"], "AF4": ["AF3", "F4"],
        "F7": ["F3", "FC5", "T7"], "F3": ["AF3", "F7", "FC5"],
        "F4": ["AF4", "F8", "FC6"], "F8": ["F4", "FC6", "T8"],
        "FC5": ["F7", "F3", "T7"], "FC6": ["F4", "F8", "T8"],
        "T7": ["F7", "FC5", "P7"], "T8": ["F8", "FC6", "P8"],
        "P7": ["T7"], "P8": ["T8"], "O1": ["O2"], "O2": ["O1"],
    }
    eav_edges = {
        "Fp1": ["F7", "F3", "Fp2"],
        "Fp2": ["Fp1", "F4", "F8"],
        "F7": ["Fp1", "F3", "FC5", "T7"],
        "F3": ["Fp1", "F7", "Fz", "FC1", "C3"],
        "Fz": ["F3", "F4", "FC1", "FC2", "Cz"],
        "F4": ["Fp2", "Fz", "F8", "FC2", "C4"],
        "F8": ["Fp2", "F4", "FC6", "T8"],
        "FC5": ["F7", "FC1", "C3", "T7"],
        "FC1": ["F3", "Fz", "FC5", "FC2", "C3", "Cz"],
        "FC2": ["Fz", "F4", "FC1", "FC6", "Cz", "C4"],
        "FC6": ["F8", "FC2", "C4", "T8"],
        "T7": ["F7", "FC5", "C3", "CP5", "P7"],
        "C3": ["FC5", "FC1", "T7", "Cz", "CP1", "CP5"],
        "Cz": ["FC1", "FC2", "C3", "C4", "CP1", "CP2"],
        "C4": ["FC2", "FC6", "Cz", "T8", "CP2", "CP6"],
        "T8": ["F8", "FC6", "C4", "CP6", "P8"],
        "CP5": ["T7", "C3", "CP1", "P3", "P7"],
        "CP1": ["C3", "Cz", "CP5", "CP2", "P3", "Pz"],
        "CP2": ["Cz", "C4", "CP1", "CP6", "Pz", "P4"],
        "CP6": ["C4", "T8", "CP2", "P4", "P8"],
        "P7": ["T7", "CP5", "P3", "PO9"],
        "P3": ["CP5", "CP1", "P7", "Pz", "O1"],
        "Pz": ["CP1", "CP2", "P3", "P4", "Oz"],
        "P4": ["CP2", "CP6", "Pz", "P8", "O2"],
        "P8": ["T8", "CP6", "P4", "PO10"],
        "PO9": ["P7", "O1"],
        "O1": ["PO9", "P3", "Oz"],
        "Oz": ["O1", "O2", "Pz"],
        "O2": ["Oz", "P4", "PO10"],
        "PO10": ["P8", "O2"],
    }

    if dataset_key in {"deap", "mahnob", "mahnob-hci"}:
        return _adjacency_from_edges(CHANNEL_ORDERS[dataset_key], deap_mahnob_edges)
    if dataset_key == "amigos":
        return _adjacency_from_edges(CHANNEL_ORDERS["amigos"], amigos_edges)
    if dataset_key == "eav":
        return _adjacency_from_edges(CHANNEL_ORDERS["eav"], eav_edges)

    if num_electrodes == 32:
        return _adjacency_from_edges(CHANNEL_ORDERS["deap"], deap_mahnob_edges)
    if num_electrodes == 14:
        return _adjacency_from_edges(CHANNEL_ORDERS["amigos"], amigos_edges)
    if num_electrodes == 30:
        return _adjacency_from_edges(CHANNEL_ORDERS["eav"], eav_edges)
    return torch.eye(num_electrodes, dtype=torch.float32)


class BiasReLU(nn.Module):
    def __init__(self, bias_shape: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.bias)


class ChebyshevGraphConvolution(nn.Module):
    """Chebyshev graph convolution."""

    def __init__(self, k: int, in_channels: int, out_channels: int):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_polynomial(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        t0 = x
        if self.k == 1:
            return t0.unsqueeze(1)

        terms = [t0, torch.matmul(lap, x)]
        for _ in range(2, self.k):
            terms.append(2 * torch.matmul(lap, terms[-1]) - terms[-2])
        return torch.stack(terms[: self.k], dim=1)

    def forward(self, x: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        cheby = self.chebyshev_polynomial(x, lap)
        cheby = cheby.permute(0, 2, 3, 1).flatten(start_dim=2)
        return torch.matmul(cheby, self.weight)


class EEGEncoder(nn.Module):
    """EEG encoder for input shape (B, C, 5)."""

    def __init__(
        self,
        num_electrodes: int = 32,
        in_channels: int = 5,
        d_g: int = 64,
        output_dim: int = 768,
        k_graph: int = 2,
        dataset_name: Optional[str] = None,
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels

        self.input_bn = nn.BatchNorm1d(in_channels)

        adj_prior = default_eeg_adjacency(num_electrodes, dataset_name=dataset_name)
        self.register_buffer("adj_mask", (adj_prior > 0).float())
        self.adj = nn.Parameter(adj_prior.clone())
        self.adj_bias = nn.Parameter(torch.zeros(1))

        self.gcn = ChebyshevGraphConvolution(k_graph, in_channels, d_g)
        self.b_relu = BiasReLU(d_g)

        self.conv_k1 = nn.Conv1d(d_g, d_g, kernel_size=1, padding="same")
        self.conv_k3 = nn.Conv1d(d_g, d_g, kernel_size=3, padding="same")
        self.conv_k5 = nn.Conv1d(d_g, d_g, kernel_size=5, padding="same")
        self.proj = nn.Linear(4 * d_g, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != self.num_electrodes or x.size(2) != self.in_channels:
            raise ValueError(
                f"Expected EEG input shape (B, {self.num_electrodes}, {self.in_channels}), "
                f"got {tuple(x.shape)}. CAMER expects C x 5 DE features."
            )

        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        adj = F.relu(self.adj + self.adj_bias) * self.adj_mask
        lap = normalized_laplacian(adj)
        g = self.b_relu(self.gcn(x, lap))

        f0 = g.permute(0, 2, 1)
        f1 = F.relu(self.conv_k1(f0))
        f2 = F.relu(self.conv_k3(f1))
        f3 = F.relu(self.conv_k5(f2))

        fused = torch.cat([f0, f1, f2, f3], dim=1).permute(0, 2, 1)
        return F.relu(self.proj(fused))


class VideoEncoder(nn.Module):
    """Video encoder for input shape (B, T, H, W, 3)."""

    def __init__(
        self,
        input_size: int = 768,
        num_select: int = 1,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        weights_root: Optional[str] = "weights",
        train_swin: bool = True,
        image_processor: Optional[AutoImageProcessor] = None,
        swin_model: Optional[SwinModel] = None,
    ):
        super().__init__()
        self.num_select = num_select
        self.train_swin = train_swin

        self.img_processor, self.swin_model = self._build_backbone(
            pretrained_model=pretrained_model,
            weights_root=weights_root,
            image_processor=image_processor,
            swin_model=swin_model,
        )
        for param in self.swin_model.parameters():
            param.requires_grad = train_swin

        self._register_image_normalization()
        self.topk_value_proj = nn.Linear(input_size, input_size)
        self.topk_weight_proj = nn.Linear(input_size, 1)

    @staticmethod
    def _build_backbone(
        pretrained_model: str,
        weights_root: Optional[str],
        image_processor: Optional[AutoImageProcessor],
        swin_model: Optional[SwinModel],
    ):
        if image_processor is not None and swin_model is not None:
            return image_processor, swin_model

        model_path = None
        if weights_root:
            candidate = Path(weights_root) / pretrained_model
            if candidate.is_dir():
                model_path = str(candidate)

        load_name = model_path or pretrained_model
        processor = image_processor or AutoImageProcessor.from_pretrained(load_name)
        model = swin_model or SwinModel.from_pretrained(load_name)
        if hasattr(processor, "do_rescale"):
            processor.do_rescale = False
        return processor, model

    def _register_image_normalization(self) -> None:
        if bool(getattr(self.img_processor, "do_normalize", True)):
            mean = getattr(self.img_processor, "image_mean", [0.485, 0.456, 0.406])
            std = getattr(self.img_processor, "image_std", [0.229, 0.224, 0.225])
            mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
            std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        else:
            mean_tensor = torch.zeros(1, 3, 1, 1)
            std_tensor = torch.ones(1, 3, 1, 1)

        self.register_buffer("img_mean", mean_tensor, persistent=False)
        self.register_buffer("img_std", std_tensor, persistent=False)

    def _mil_selection(self, patch_tokens: torch.Tensor, pooled_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_patches, hidden_dim = patch_tokens.shape
        hidden = torch.tanh(self.topk_value_proj(pooled_tokens))
        weights = F.softmax(self.topk_weight_proj(hidden), dim=1)
        indices = torch.topk(weights, self.num_select, dim=1).indices.squeeze(-1)

        gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            self.num_select,
            num_patches,
            hidden_dim,
        )
        selected_patches = patch_tokens.gather(dim=1, index=gather_index)
        return selected_patches.contiguous().view(batch_size, -1, hidden_dim)

    def forward(self, images_data: torch.Tensor) -> torch.Tensor:
        if images_data.dim() != 5 or images_data.size(-1) != 3:
            raise ValueError(
                f"Expected image input shape (B, T, H, W, 3), got {tuple(images_data.shape)}."
            )

        batch_size = images_data.size(0)
        num_frames = images_data.size(1)
        pixel_values = images_data.reshape(
            batch_size * num_frames,
            images_data.size(2),
            images_data.size(3),
            3,
        ).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values.float()
        pixel_values = (pixel_values - self.img_mean.to(pixel_values.dtype)) / self.img_std.to(pixel_values.dtype)

        if self.train_swin:
            visual_output = self.swin_model(pixel_values=pixel_values)
        else:
            with torch.no_grad():
                visual_output = self.swin_model(pixel_values=pixel_values)

        patch_tokens = visual_output.last_hidden_state
        num_patches = patch_tokens.size(1)
        hidden_dim = patch_tokens.size(2)
        patch_tokens = patch_tokens.view(batch_size, num_frames, num_patches, hidden_dim)
        pooled_tokens = patch_tokens.mean(dim=2)
        return self._mil_selection(patch_tokens, pooled_tokens)


class CrossAttention(nn.Module):
    """Bidirectional EEG-video cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int = 12, dropout: float = 0.2):
        super().__init__()
        self.attn_e2v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_v2e = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, feat_e: torch.Tensor, feat_v: torch.Tensor):
        feat_e_prime, _ = self.attn_e2v(query=feat_e, key=feat_v, value=feat_v)
        feat_v_prime, _ = self.attn_v2e(query=feat_v, key=feat_e, value=feat_e)
        return feat_e_prime, feat_v_prime


class SupConLoss(nn.Module):
    """Bidirectional supervised contrastive loss."""

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5, beta: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def _directional_loss(
        self,
        z_anchor: torch.Tensor,
        z_pool: torch.Tensor,
        labels_anchor: torch.Tensor,
        labels_pool: torch.Tensor,
        subject_anchor: torch.Tensor,
        subject_pool: torch.Tensor,
    ) -> torch.Tensor:
        z_anchor = F.normalize(z_anchor, dim=1)
        z_pool = F.normalize(z_pool, dim=1)

        logits = torch.matmul(z_anchor, z_pool.T) / self.temperature

        labels_anchor = labels_anchor.view(-1, 1)
        labels_pool = labels_pool.view(1, -1)
        subject_anchor = subject_anchor.view(-1, 1)
        subject_pool = subject_pool.view(1, -1)

        same_label = torch.eq(labels_anchor, labels_pool).float()
        same_subject = torch.eq(subject_anchor, subject_pool).float()

        strong_pos = same_subject * same_label
        weak_pos = (1.0 - same_subject) * same_label
        hard_neg = same_subject * (1.0 - same_label)

        pos_weights = strong_pos + weak_pos * self.alpha
        neg_weights = hard_neg * self.beta + (1.0 - hard_neg)

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * neg_weights
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        weight_sum = pos_weights.sum(dim=1)
        weight_sum = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
        return -((pos_weights * log_prob).sum(dim=1) / weight_sum).mean()

    def forward(
        self,
        z_e: torch.Tensor,
        z_v: torch.Tensor,
        labels: torch.Tensor,
        subjects: torch.Tensor,
        labels_v: Optional[torch.Tensor] = None,
        subjects_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        labels_v = labels if labels_v is None else labels_v
        subjects_v = subjects if subjects_v is None else subjects_v

        loss_e2v = self._directional_loss(z_e, z_v, labels, labels_v, subjects, subjects_v)
        loss_v2e = self._directional_loss(z_v, z_e, labels_v, labels, subjects_v, subjects)
        return 0.5 * (loss_e2v + loss_v2e)


class CAMERModel(nn.Module):
    """CAMER multimodal emotion recognition model."""

    def __init__(
        self,
        input_size: int = 768,
        num_classes: int = 4,
        num_heads: int = 12,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 2,
        num_electrodes: int = 32,
        eeg_size: int = 5,
        dataset_name: Optional[str] = None,
        num_select: int = 1,
        transformer_dropout_rate: float = 0.2,
        cls_dropout_rate: float = 0.1,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        weights_root: Optional[str] = "weights",
        train_swin: bool = True,
        use_cross_attn: bool = True,
        use_scl: bool = True,
        use_mmt: bool = True,
        image_processor: Optional[AutoImageProcessor] = None,
        swin_model: Optional[SwinModel] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_select = num_select
        self.use_cross_attn = use_cross_attn
        self.use_scl = use_scl
        self.use_mmt = use_mmt

        self.video_encoder = VideoEncoder(
            input_size=input_size,
            num_select=num_select,
            pretrained_model=pretrained_model,
            weights_root=weights_root,
            train_swin=train_swin,
            image_processor=image_processor,
            swin_model=swin_model,
        )

        self.eeg_encoder = EEGEncoder(
            num_electrodes=num_electrodes,
            in_channels=eeg_size,
            d_g=64,
            output_dim=input_size,
            k_graph=2,
            dataset_name=dataset_name,
        )

        if use_cross_attn:
            self.cross_attn = CrossAttention(
                input_size,
                num_heads=num_heads,
                dropout=transformer_dropout_rate,
            )

        if use_mmt:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout_rate,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(cls_dropout_rate),
            nn.Linear(input_size, num_classes),
        )

        if use_scl:
            self.proj_e = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, input_size),
            )
            self.proj_v = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, input_size),
            )

    def forward(self, eeg_data: torch.Tensor, images_data: torch.Tensor):
        batch_size = eeg_data.size(0)
        he = self.eeg_encoder(eeg_data)
        hv = self.video_encoder(images_data)

        z_e, z_v = None, None
        if self.use_scl:
            z_e = F.normalize(self.proj_e(he.mean(dim=1)), dim=1)
            z_v = F.normalize(self.proj_v(hv.mean(dim=1)), dim=1)

        tokens = [hv, he]
        if self.use_cross_attn:
            he_prime, hv_prime = self.cross_attn(he, hv)
            tokens.extend([hv_prime, he_prime])

        if not self.use_mmt:
            logits = self.classifier(torch.cat(tokens, dim=1).mean(dim=1))
            return logits, z_e, z_v

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_token, *tokens], dim=1)
        seq_out = self.transformer_encoder(seq)
        logits = self.classifier(seq_out[:, 0, :])
        return logits, z_e, z_v


def build_camer_model(dataset_name: str = "deap", **overrides) -> CAMERModel:
    """Create CAMERModel from parameters.py defaults."""
    return CAMERModel(**build_model_params(dataset_name, **overrides))


def build_scl_loss(**overrides) -> SupConLoss:
    """Create SupConLoss from parameters.py defaults."""
    params = dict(SCL_PARAMS)
    params.update({key: value for key, value in overrides.items() if value is not None})
    return SupConLoss(**params)


__all__ = [
    "BiasReLU",
    "CAMERModel",
    "ChebyshevGraphConvolution",
    "CrossAttention",
    "EEGEncoder",
    "SupConLoss",
    "VideoEncoder",
    "build_camer_model",
    "build_scl_loss",
    "default_eeg_adjacency",
    "normalized_laplacian",
]
