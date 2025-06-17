from __future__ import annotations

import abc
from pathlib import Path
from typing import Tuple, Dict

import tensorflow as tf
import numpy as np


class BaseDataset(abc.ABC):
    """Abstract helper that turns NumPy arrays into batched tf.data pipelines."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        val_split: float = 0.1,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # Check dataset 
        labelled_dir = self.data_dir / "labelled"
        unlabelled_dir = self.data_dir / "unlabelled"

        if (not labelled_dir.exists()) and (not unlabelled_dir.exists()):
            print(
                "Warning: No labelled or unlabelled directories found. "
                "Setting up dataset..."
            )
            self.setup()

        if not labelled_dir.exists() or not labelled_dir.is_dir():
            raise FileNotFoundError(f"Labelled directory not found: {labelled_dir}")
        if not unlabelled_dir.exists() or not unlabelled_dir.is_dir():
            raise FileNotFoundError(f"Unlabelled directory not found: {unlabelled_dir}")

        # Check that labelled contains at least one class folder
        self.class_names = [f for f in labelled_dir.iterdir() if f.is_dir()]
        if not self.class_names:
            raise FileNotFoundError(f"No class folders found in labelled directory: {labelled_dir}")

        # Check that unlabelled contains at least one image file
        image_files = list(unlabelled_dir.glob("*"))
        if not any(f.is_file() for f in image_files):
            raise FileNotFoundError(f"No image files found in unlabelled directory: {unlabelled_dir}")
        
        # Show class and total files info
        print(f"Found {len(self.class_names)} class folders in labelled directory: {labelled_dir}")
        print(f"Found {len(image_files)} image files in unlabelled directory: {unlabelled_dir}")

        self.num_classes = len(self.class_names)
        self.labelled = ([], [])
        self.unlabelled = ([], [])

    @abc.abstractmethod
    def setup(self):
        raise NotImplementedError("Dataset folder should contains \"labelled\" and \"unlabelled\" directories or dataset setup not implemented.")
    
    @abc.abstractmethod
    def build(self):
        raise NotImplementedError("Dataset build is not implemented!")

    @abc.abstractmethod
    def make_iter(self, subset, shuffle=True):
        raise NotImplementedError("Dataset build is not implemented!")
    
    # # --------------------------------------------------------------------- API
    # def load(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    #     """Return (train_ds, val_ds, test_ds)."""
    #     (x_train, y_train), (x_test, y_test) = self._load_raw()

    #     # deterministic train/val split
    #     n_val = int(len(x_train) * self.val_split)
    #     rng = np.random.default_rng(self.seed)
    #     idx = rng.permutation(len(x_train))
    #     val_idx, train_idx = idx[:n_val], idx[n_val:]

    #     x_val, y_val = x_train[val_idx], y_train[val_idx]
    #     x_train, y_train = x_train[train_idx], y_train[train_idx]

    #     train_ds = self._make_dataset(x_train, y_train, training=True)
    #     val_ds = self._make_dataset(x_val, y_val, training=False)
    #     test_ds = self._make_dataset(x_test, y_test, training=False)
    #     return train_ds, val_ds, test_ds

    # # -----------------------------------------------------------------
    # @abc.abstractmethod
    # def _load_raw(self):
    #     """Return raw (x_train, y_train), (x_test, y_test) NumPy arrays."""
    #     ...

    # # ----------------------------------------------------------------- helpers
    # def _preprocess(self, x, y):
    #     x = tf.cast(x, tf.float32) / 255.0
    #     return x, y

    # def _augment(self, x, y):
    #     if self.input_shape[-1] == 3:  # RGB only
    #         x = tf.image.random_flip_left_right(x)
    #         x = tf.image.random_rotation(x, 0.1)
    #     return x, y

    # def _make_dataset(self, x, y, training: bool):
    #     ds = tf.data.Dataset.from_tensor_slices((x, y))
    #     if training:
    #         ds = ds.shuffle(self.shuffle_buffer, seed=self.seed)
    #     ds = ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    #     if training and self.augment:
    #         ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
    #     return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
