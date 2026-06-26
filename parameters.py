# -*- coding: utf-8 -*-
"""Default hyperparameters for CAMER."""


DATASET_SPECS = {
    "deap": {"num_electrodes": 32, "num_classes": 4},
    "mahnob": {"num_electrodes": 32, "num_classes": 4},
    "mahnob-hci": {"num_electrodes": 32, "num_classes": 4},
    "amigos": {"num_electrodes": 14, "num_classes": 4},
    "eav": {"num_electrodes": 30, "num_classes": 5},
}


DEAP_MAHNOB_CHANNEL_ORDER = [
    "FP1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "OZ", "PZ",
    "FP2", "AF4", "FZ", "F4", "F8", "FC6", "FC2", "CZ",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]

AMIGOS_CHANNEL_ORDER = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

EAV_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "PO9", "O1", "Oz", "O2", "PO10",
]

CHANNEL_ORDERS = {
    "deap": DEAP_MAHNOB_CHANNEL_ORDER,
    "mahnob": DEAP_MAHNOB_CHANNEL_ORDER,
    "mahnob-hci": DEAP_MAHNOB_CHANNEL_ORDER,
    "amigos": AMIGOS_CHANNEL_ORDER,
    "eav": EAV_CHANNEL_ORDER,
}


# EEG input representation: five-band DE features with shape C x 5.
DATA_PARAMS = {
    "eeg_feature": "five_band_de",
    "eeg_size": 5,
    "num_instances": 4,
}


MODEL_PARAMS = {
    # Visual encoder.
    "pretrained_model": "microsoft/swin-tiny-patch4-window7-224",
    "weights_root": "weights",
    "train_swin": True,

    # Shared token dimension and EEG feature size.
    "input_size": 768,
    "eeg_size": 5,

    # Number of selected key frames in the MIL module.
    "num_select": 1,

    # Multimodal Transformer.
    "num_heads": 12,
    "dim_feedforward": 2048,
    "num_encoder_layers": 2,
    "transformer_dropout_rate": 0.2,
    "cls_dropout_rate": 0.1,

    # Module switches.
    "use_scl": True,
    "use_cross_attn": True,
    "use_mmt": True,
}


SCL_PARAMS = {
    # Structured supervised contrastive loss.
    "temperature": 0.07,
    "alpha": 0.5,
    "beta": 2.0,
}


TRAINING_PARAMS = {
    # Optimization and evaluation settings.
    "epochs": 50,
    "batch_size": 64,
    "optimizer": "AdamW",
    "lr": 1e-4,
    "swin_lr": 1e-4,
    "weight_decay": 0.01,
    "scheduler": "CosineAnnealingLR",

    # Loss trade-off in L = (1 - lambda) * LCE + lambda * LSCL.
    "lambda_scl": 0.5,

    "metrics": ["accuracy", "macro_f1"],
    "evaluation_protocol": "five_fold_cross_trial",
}


def get_dataset_spec(dataset_name):
    """Return default electrode and class counts for a dataset."""
    name = dataset_name.lower()
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dict(DATASET_SPECS[name])


def build_model_params(dataset_name="deap", **overrides):
    """Build keyword arguments for CAMERModel."""
    name = dataset_name.lower()
    spec = get_dataset_spec(dataset_name)
    params = dict(MODEL_PARAMS)
    params.update(
        {
            "dataset_name": name,
            "num_electrodes": spec["num_electrodes"],
            "num_classes": spec["num_classes"],
        }
    )
    params.update({key: value for key, value in overrides.items() if value is not None})
    return params


__all__ = [
    "CHANNEL_ORDERS",
    "AMIGOS_CHANNEL_ORDER",
    "DEAP_MAHNOB_CHANNEL_ORDER",
    "EAV_CHANNEL_ORDER",
    "DATASET_SPECS",
    "DATA_PARAMS",
    "MODEL_PARAMS",
    "SCL_PARAMS",
    "TRAINING_PARAMS",
    "build_model_params",
    "get_dataset_spec",
]
