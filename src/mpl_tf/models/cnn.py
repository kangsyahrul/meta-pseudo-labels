from __future__ import annotations
from typing import Sequence, Tuple

import tensorflow as tf
from .base import BaseModel


class CNN(BaseModel):
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        conv_filters: Sequence[int] = (64, 32),
        dense_units: int = 128,
        dropout_rate: float = 0.1,
        **compile_kwargs,
    ) -> None:
        super().__init__(compile_kwargs=compile_kwargs)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        # Input layer
        inputs = tf.keras.layers.Input(shape=self.input_shape, name="input")

        # Convolutional blocks
        x = inputs
        for block_idx, filters in enumerate(self.conv_filters, start=1):
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=3,
                activation="relu",
                padding="same",
                name=f"conv{block_idx}",
            )(x)
            x = tf.keras.layers.MaxPool2D(name=f"pool{block_idx}")(x)

        # Classifier head
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(self.dense_units, name="dense")(x)
        x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
        x = tf.keras.layers.ReLU(name="relu")(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name="dropout")(x)
        logits = tf.keras.layers.Dense(self.num_classes, name="logits")(x)

        # Create and compile the model
        model = tf.keras.Model(inputs, logits, name="mnist_mpl")
        model.compile(**self.compile_kwargs)
        return model

