import os
import yaml
import numpy as np
import torch
# Work around PyTorch 2.x Dynamo/inspect interaction with SpeechBrain's
# lazy k2 module on this Windows/Python 3.12 environment.
import torch.optim.optimizer as _torch_optimizer

if hasattr(_torch_optimizer.Optimizer.add_param_group, "__wrapped__"):
    _torch_optimizer.Optimizer.add_param_group = (
        _torch_optimizer.Optimizer.add_param_group.__wrapped__
    )

if hasattr(_torch_optimizer.Optimizer.zero_grad, "__wrapped__"):
    _torch_optimizer.Optimizer.zero_grad = (
        _torch_optimizer.Optimizer.zero_grad.__wrapped__
    )

from openwakeword.train import Model


def main():
    config_path = os.path.join(
        os.path.dirname(__file__),
        "asta_training.yml"
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("ASTA OPENWAKEWORD TRAINING")
    print("=" * 70)

    positive_path = config["feature_data_files"]["positive"]
    negative_path = config["feature_data_files"]["adversarial_negative"]
    fp_validation_path = config["false_positive_validation_data_path"]

    print(f"Positive features : {positive_path}")
    print(f"Negative features : {negative_path}")
    print(f"FP validation     : {fp_validation_path}")

    # Memory-map the feature arrays so we don't load 5.6M negatives
    # into RAM all at once.
    positive = np.load(positive_path, mmap_mode="r")
    negative = np.load(negative_path, mmap_mode="r")

    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")

    if positive.shape[1:] != negative.shape[1:]:
        raise ValueError(
            f"Feature shape mismatch: "
            f"{positive.shape[1:]} vs {negative.shape[1:]}"
        )

    input_shape = positive.shape[1:]

    print(f"Input shape: {input_shape}")

    # Create model.
    oww = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type=config["model_type"],
        layer_dim=config["layer_size"],
        seconds_per_example=1280 * input_shape[0] / 16000,
    )

    # Labels:
    # positive = 1
    # negative = 0
    #
    # The existing ACAV100M feature file is intentionally memory-mapped.
    feature_data_files = {
        "positive": positive_path,
        "negative": negative_path,
    }

    data_transforms = {
        "positive": lambda x: x,
        "negative": lambda x: x,
    }

    label_transforms = {
        "positive": lambda x: [1 for _ in x],
        "negative": lambda x: [0 for _ in x],
    }

    # Use the same mmap batch generator used by openWakeWord's
    # training implementation.
    from openwakeword.train import mmap_batch_generator

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

    # Keep this conservative on Windows.
    X_train = torch.utils.data.DataLoader(
        IterDataset(batch_generator),
        batch_size=None,
        num_workers=0,
    )

    # False-positive validation data.
    X_val_fp_raw = np.load(fp_validation_path, mmap_mode="r")

    print(f"FP validation raw shape: {X_val_fp_raw.shape}")

# validation_set_features.npy contains individual 96-feature frames.
# The wakeword model consumes 16 consecutive frames per example.
    n_windows = X_val_fp_raw.shape[0] // input_shape[0]

    X_val_fp_data = X_val_fp_raw[:n_windows * input_shape[0]].reshape(
        n_windows,
        input_shape[0],
        input_shape[1],
    )

    print(f"FP validation windowed shape: {X_val_fp_data.shape}")

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

    # Balanced positive/negative validation set.
    # Use a subset of the huge negative dataset rather than loading
    # millions of examples into RAM.
    n_val_pos = min(positive.shape[0], 2500)
    n_val_neg = n_val_pos

    pos_val = np.asarray(positive[:n_val_pos])
    neg_val = np.asarray(negative[:n_val_neg])

    X_val_data = np.vstack((pos_val, neg_val))
    X_val_labels = np.hstack((
        np.ones(n_val_pos),
        np.zeros(n_val_neg),
    )).astype(np.float32)

    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_data),
            torch.from_numpy(X_val_labels),
        ),
        batch_size=len(X_val_labels),
    )

    print(f"Balanced validation shape: {X_val_data.shape}")

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Steps: {config['steps']}")
    print(f"Max negative weight: {config['max_negative_weight']}")
    print(
        "Target false positives/hour: "
        f"{config['target_false_positives_per_hour']}"
    )

    # Train using the same auto-training entry point as the
    # notebook/local openWakeWord training implementation.
    best_model = oww.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=config["steps"],
        max_negative_weight=config["max_negative_weight"],
        target_fp_per_hour=config["target_false_positives_per_hour"],
    )

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("EXPORTING ONNX MODEL")
    print("=" * 70)

    oww.export_model(
        model=best_model,
        model_name=config["model_name"],
        output_dir=output_dir,
    )

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()