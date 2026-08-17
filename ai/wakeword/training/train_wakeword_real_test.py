import argparse
import os

import numpy as np
import torch
import yaml

# ------------------------------------------------------------------
# PyTorch workaround
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

from openwakeword.train import Model


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
# Constants
# ------------------------------------------------------------------

ALL_PHRASES = [
    "hello_asta",
    "hey_asta",
    "wake_up_asta",
]

HARD_NEGATIVE_FRACTION = 0.25

RANDOM_SEED = 42


# ------------------------------------------------------------------
# Custom mixed mmap generator
# ------------------------------------------------------------------

class MixedNegativeBatchGenerator:

    """
    Generates balanced positive/negative batches.

    Positive:
        target wake phrase

    Negative:
        mostly ACAV100M
        plus phrase-specific hard negatives

    Hard negatives are sampled from the OTHER two wake phrases.
    """

    def __init__(
        self,
        positive_path,
        acav_negative_path,
        hard_negative_paths,
        n_positive,
        n_negative,
        hard_negative_fraction=0.25,
        seed=42,
    ):

        self.positive = np.load(
            positive_path,
            mmap_mode="r",
        )

        self.acav_negative = np.load(
            acav_negative_path,
            mmap_mode="r",
        )

        self.hard_negatives = [
            np.load(
                path,
                mmap_mode="r",
            )
            for path in hard_negative_paths
        ]

        self.n_positive = int(n_positive)
        self.n_negative = int(n_negative)

        self.hard_negative_fraction = (
            hard_negative_fraction
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.positive_counter = 0
        self.acav_counter = 0

        self.hard_counters = [
            0 for _ in self.hard_negatives
        ]

        self.hard_counter = 0

    def _next_positive(self):

        start = self.positive_counter

        end = start + self.n_positive

        if end <= self.positive.shape[0]:

            x = self.positive[
                start:end
            ]

            self.positive_counter = end

        else:

            first = self.positive[
                start:
            ]

            remaining = (
                self.n_positive
                - len(first)
            )

            second = self.positive[
                :remaining
            ]

            x = np.concatenate(
                [first, second],
                axis=0,
            )

            self.positive_counter = remaining

        return np.asarray(
            x,
            dtype=np.float32,
        )

    def _next_acav(self, count):

        if count <= 0:
            return np.empty(
                (
                    0,
                    self.acav_negative.shape[1],
                    self.acav_negative.shape[2],
                ),
                dtype=np.float32,
            )

        # Random sampling is preferable here because
        # ACAV100M is enormous and we don't want the training
        # batches to walk through only one small sequential region.
        indices = self.rng.integers(
            0,
            self.acav_negative.shape[0],
            size=count,
        )

        return np.asarray(
            self.acav_negative[indices],
            dtype=np.float32,
        )

    def _next_hard(self, count):

        if count <= 0:

            return np.empty(
                (
                    0,
                    self.positive.shape[1],
                    self.positive.shape[2],
                ),
                dtype=np.float32,
            )

        result = []

        # Rotate between the two wrong wake phrases.
        for _ in range(count):

            source_index = (
                self.hard_counter
                % len(self.hard_negatives)
            )

            source = self.hard_negatives[
                source_index
            ]

            index = self.rng.integers(
                0,
                source.shape[0],
            )

            result.append(
                np.asarray(
                    source[index],
                    dtype=np.float32,
                )
            )

            self.hard_counter += 1

        return np.stack(
            result,
            axis=0,
        )

    def __iter__(self):
        return self

    def __next__(self):

        # ----------------------------------------------------------
        # Positive examples
        # ----------------------------------------------------------

        positive = self._next_positive()

        # ----------------------------------------------------------
        # Negative examples
        # ----------------------------------------------------------

        n_hard = int(
            round(
                self.n_negative
                * self.hard_negative_fraction
            )
        )

        n_acav = (
            self.n_negative
            - n_hard
        )

        acav = self._next_acav(
            n_acav
        )

        hard = self._next_hard(
            n_hard
        )

        negative = np.concatenate(
            [
                acav,
                hard,
            ],
            axis=0,
        )

        # Shuffle negative examples so the model
        # doesn't see ACAV and hard negatives in
        # fixed blocks.
        self.rng.shuffle(
            negative,
            axis=0,
        )

        # ----------------------------------------------------------
        # Build batch
        # ----------------------------------------------------------

        X = np.concatenate(
            [
                positive,
                negative,
            ],
            axis=0,
        )

        y = np.concatenate(
            [
                np.ones(
                    len(positive),
                    dtype=np.float32,
                ),
                np.zeros(
                    len(negative),
                    dtype=np.float32,
                ),
            ]
        )

        # Shuffle complete batch.
        indices = self.rng.permutation(
            len(X)
        )

        X = X[indices]
        y = y[indices]

        return X, y


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def get_phrase_feature_path(
    feature_dir,
    phrase,
):

    return os.path.join(
        feature_dir,
        f"{phrase}_real_only_features.npy",
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    config_path = os.path.join(
        os.path.dirname(__file__),
        "asta_training.yml",
    )

    with open(
        config_path,
        "r",
    ) as f:

        config = yaml.safe_load(f)

    # --------------------------------------------------------------
    # Feature directory
    # --------------------------------------------------------------

    positive_config_path = config[
        "feature_data_files"
    ][
        "positive"
    ]

    feature_dir = os.path.dirname(
        positive_config_path
    )

    # --------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------

    positive_path = (
        get_phrase_feature_path(
            feature_dir,
            PHRASE,
        )
    )

    negative_path = config[
        "feature_data_files"
    ][
        "adversarial_negative"
    ]

    fp_validation_path = config[
        "false_positive_validation_data_path"
    ]

    hard_negative_phrases = [
        phrase
        for phrase in ALL_PHRASES
        if phrase != PHRASE
    ]

    hard_negative_paths = [
        get_phrase_feature_path(
            feature_dir,
            phrase,
        )
        for phrase in hard_negative_phrases
    ]

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    print("=" * 70)
    print(
        f"TRAINING WAKEWORD: {PHRASE}"
    )
    print("=" * 70)

    print(
        "Positive:",
        positive_path,
    )

    print(
        "ACAV negative:",
        negative_path,
    )

    print(
        "Hard negatives:",
    )

    for path in hard_negative_paths:
        print(
            "  ",
            path,
        )

    # --------------------------------------------------------------
    # Load feature arrays
    # --------------------------------------------------------------

    positive = np.load(
        positive_path,
        mmap_mode="r",
    )

    negative = np.load(
        negative_path,
        mmap_mode="r",
    )

    hard_negative_arrays = [
        np.load(
            path,
            mmap_mode="r",
        )
        for path in hard_negative_paths
    ]

    print(
        "Positive shape:",
        positive.shape,
    )

    print(
        "ACAV negative shape:",
        negative.shape,
    )

    for phrase, array in zip(
        hard_negative_phrases,
        hard_negative_arrays,
    ):

        print(
            f"{phrase} hard negative shape:",
            array.shape,
        )

    # --------------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------------

    input_shape = positive.shape[1:]

    if negative.shape[1:] != input_shape:

        raise ValueError(
            "ACAV negative feature mismatch: "
            f"{negative.shape[1:]} vs "
            f"{input_shape}"
        )

    for phrase, array in zip(
        hard_negative_phrases,
        hard_negative_arrays,
    ):

        if array.shape[1:] != input_shape:

            raise ValueError(
                f"Hard negative feature mismatch "
                f"for {phrase}: "
                f"{array.shape[1:]} vs "
                f"{input_shape}"
            )

    # --------------------------------------------------------------
    # Determine batch sizes
    # --------------------------------------------------------------

    configured_n = config[
        "batch_n_per_class"
    ]

    if isinstance(
        configured_n,
        dict,
    ):

        n_positive = int(
            configured_n.get(
                "1",
                configured_n.get(
                    1,
                    32,
                ),
            )
        )

        n_negative = int(
            configured_n.get(
                "0",
                configured_n.get(
                    0,
                    32,
                ),
            )
        )

    else:

        n_positive = int(
            configured_n
        )

        n_negative = int(
            configured_n
        )

    print()
    print(
        f"Positive examples/batch: "
        f"{n_positive}"
    )

    print(
        f"Negative examples/batch: "
        f"{n_negative}"
    )

    print(
        f"Hard-negative fraction: "
        f"{HARD_NEGATIVE_FRACTION:.0%}"
    )

    print(
        f"Hard negatives/batch: "
        f"{round(n_negative * HARD_NEGATIVE_FRACTION)}"
    )

    print(
        f"ACAV negatives/batch: "
        f"{n_negative - round(n_negative * HARD_NEGATIVE_FRACTION)}"
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type=config["model_type"],
        layer_dim=config["layer_size"],
        seconds_per_example=(
            1280
            * input_shape[0]
            / 16000
        ),
    )

    # --------------------------------------------------------------
    # Training generator
    # --------------------------------------------------------------

    batch_generator = MixedNegativeBatchGenerator(
        positive_path=positive_path,
        acav_negative_path=negative_path,
        hard_negative_paths=hard_negative_paths,
        n_positive=n_positive,
        n_negative=n_negative,
        hard_negative_fraction=(
            HARD_NEGATIVE_FRACTION
        ),
        seed=RANDOM_SEED,
    )

    class IterDataset(
        torch.utils.data.IterableDataset
    ):

        def __init__(
            self,
            generator,
        ):

            self.generator = generator

        def __iter__(self):

            return self.generator

    X_train = torch.utils.data.DataLoader(
        IterDataset(
            batch_generator
        ),
        batch_size=None,
        num_workers=0,
    )

    # --------------------------------------------------------------
    # False-positive validation
    # --------------------------------------------------------------

    X_val_fp_raw = np.load(
        fp_validation_path,
        mmap_mode="r",
    )

    n_windows = (
        X_val_fp_raw.shape[0]
        // input_shape[0]
    )

    X_val_fp_data = X_val_fp_raw[
        :n_windows * input_shape[0]
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
            torch.from_numpy(
                np.array(
                    X_val_fp_data,
                    copy=True,
                )
            ),
            torch.from_numpy(
                X_val_fp_labels
            ),
        ),
        batch_size=128,
        shuffle=False,
    )

    # --------------------------------------------------------------
    # Balanced validation
    #
    # Positive:
    #     target phrase
    #
    # Negative:
    #     half ACAV
    #     half wrong wake phrases
    # --------------------------------------------------------------

    n_val_pos = min(
        positive.shape[0],
        2500,
    )

    pos_val = np.asarray(
        positive[
            :n_val_pos
        ]
    )

    # Half of negative validation comes from ACAV.
    n_val_acav = n_val_pos // 2

    # Other half comes from the two wrong wake phrases.
    n_val_hard = (
        n_val_pos
        - n_val_acav
    )

    acav_indices = np.random.default_rng(
        RANDOM_SEED
    ).integers(
        0,
        negative.shape[0],
        size=n_val_acav,
    )

    neg_val_acav = np.asarray(
        negative[
            acav_indices
        ]
    )

    # Sample hard negatives evenly from the
    # two other wake phrases.
    hard_parts = []

    hard_rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    for i in range(
        n_val_hard
    ):

        source = hard_negative_arrays[
            i % len(
                hard_negative_arrays
            )
        ]

        index = hard_rng.integers(
            0,
            source.shape[0],
        )

        hard_parts.append(
            np.asarray(
                source[index]
            )
        )

    neg_val_hard = np.stack(
        hard_parts,
        axis=0,
    )

    neg_val = np.concatenate(
        [
            neg_val_acav,
            neg_val_hard,
        ],
        axis=0,
    )

    X_val_data = np.vstack(
        [
            pos_val,
            neg_val,
        ]
    )

    X_val_labels = np.hstack(
        [
            np.ones(
                n_val_pos,
                dtype=np.float32,
            ),
            np.zeros(
                len(neg_val),
                dtype=np.float32,
            ),
        ]
    )

    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(
                X_val_data
            ),
            torch.from_numpy(
                X_val_labels
            ),
        ),
        batch_size=len(
            X_val_labels
        ),
    )

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"Positive validation: "
        f"{n_val_pos}"
    )

    print(
        f"ACAV negative validation: "
        f"{n_val_acav}"
    )

    print(
        f"Hard negative validation: "
        f"{n_val_hard}"
    )

    print(
        f"Total validation: "
        f"{len(X_val_labels)}"
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("START TRAINING")
    print("=" * 70)

    best_model = model.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=config["steps"],
        max_negative_weight=config[
            "max_negative_weight"
        ],
        target_fp_per_hour=config[
            "target_false_positives_per_hour"
        ],
    )

    # --------------------------------------------------------------
    # Export
    # --------------------------------------------------------------

    output_dir = config[
        "output_dir"
    ]

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