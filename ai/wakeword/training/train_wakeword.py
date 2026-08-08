import argparse
import os

import numpy as np
import torch
import yaml

# ------------------------------------------------------------------
# PyTorch workaround (keep this exactly as before)
# ------------------------------------------------------------------

import torch.optim.optimizer as _torch_optimizer

if hasattr(_torch_optimizer.Optimizer.add_param_group, "__wrapped__"):
    _torch_optimizer.Optimizer.add_param_group = (
        _torch_optimizer.Optimizer.add_param_group.__wrapped__
    )

if hasattr(_torch_optimizer.Optimizer.zero_grad, "__wrapped__"):
    _torch_optimizer.Optimizer.zero_grad = (
        _torch_optimizer.Optimizer.zero_grad.__wrapped__
    )

from openwakeword.train import Model, mmap_batch_generator


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--phrase",
    required=True,
    choices=[
        "hello_asta",
        "hey_asta",
        "wake_up_asta",
    ],
)

args = parser.parse_args()

PHRASE = args.phrase


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    config_path = os.path.join(
        os.path.dirname(__file__),
        "asta_training.yml",
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------------------
    # Replace ONLY the positive feature path
    # --------------------------------------------------------------

    positive_path = os.path.join(
        os.path.dirname(config["feature_data_files"]["positive"]),
        f"{PHRASE}_features.npy",
    )

    negative_path = config["feature_data_files"]["adversarial_negative"]
    fp_validation_path = config["false_positive_validation_data_path"]

    print("=" * 70)
    print(f"TRAINING WAKEWORD: {PHRASE}")
    print("=" * 70)

    print("Positive:", positive_path)
    print("Negative:", negative_path)

    positive = np.load(
        positive_path,
        mmap_mode="r",
    )

    negative = np.load(
        negative_path,
        mmap_mode="r",
    )

    print("Positive shape:", positive.shape)
    print("Negative shape:", negative.shape)

    if positive.shape[1:] != negative.shape[1:]:
        raise ValueError(
            f"Feature mismatch: {positive.shape[1:]} vs {negative.shape[1:]}"
        )

    input_shape = positive.shape[1:]

    model = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type=config["model_type"],
        layer_dim=config["layer_size"],
        seconds_per_example=1280 * input_shape[0] / 16000,
    )

    feature_data_files = {
        "positive": positive_path,
        "negative": negative_path,
    }

    data_transforms = {
        "positive": lambda x: x,
        "negative": lambda x: x,
    }

    label_transforms = {
        "positive": lambda x: [1] * len(x),
        "negative": lambda x: [0] * len(x),
    }

    batch_generator = mmap_batch_generator(
        feature_data_files,
        n_per_class=config["batch_n_per_class"],
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
    )

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator

    X_train = torch.utils.data.DataLoader(
        IterDataset(batch_generator),
        batch_size=None,
        num_workers=0,
    )

    # --------------------------------------------------------------
    # False positive validation
    # --------------------------------------------------------------

    X_val_fp_raw = np.load(
        fp_validation_path,
        mmap_mode="r",
    )

    n_windows = X_val_fp_raw.shape[0] // input_shape[0]

    X_val_fp_data = X_val_fp_raw[
        : n_windows * input_shape[0]
    ].reshape(
        n_windows,
        input_shape[0],
        input_shape[1],
    )

    X_val_fp_labels = np.zeros(
        n_windows,
        dtype=np.float32,
    )

    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(X_val_fp_data, copy=True)),
            torch.from_numpy(X_val_fp_labels),
        ),
        batch_size=128,
        shuffle=False,
    )

    # --------------------------------------------------------------
    # Balanced validation
    # --------------------------------------------------------------

    n_val_pos = min(
        positive.shape[0],
        2500,
    )

    n_val_neg = n_val_pos

    pos_val = np.asarray(
        positive[:n_val_pos]
    )

    neg_val = np.asarray(
        negative[:n_val_neg]
    )

    X_val_data = np.vstack(
        (
            pos_val,
            neg_val,
        )
    )

    X_val_labels = np.hstack(
        (
            np.ones(n_val_pos),
            np.zeros(n_val_neg),
        )
    ).astype(np.float32)

    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_data),
            torch.from_numpy(X_val_labels),
        ),
        batch_size=len(X_val_labels),
    )

    print("=" * 70)
    print("START TRAINING")
    print("=" * 70)

    best_model = model.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=config["steps"],
        max_negative_weight=config["max_negative_weight"],
        target_fp_per_hour=config["target_false_positives_per_hour"],
    )

    output_dir = config["output_dir"]

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("=" * 70)
    print("EXPORTING")
    print("=" * 70)

    model.export_model(
        model=best_model,
        model_name=PHRASE,
        output_dir=output_dir,
    )

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()