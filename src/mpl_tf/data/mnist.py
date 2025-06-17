import os
import glob
import numpy as np
import tensorflow as tf

from .base import BaseDataset


class MNISTDataset(BaseDataset):

    def setup(self):
        # Load MNIST dataset
        print(f"Downloading dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Create target directories for each class if they don't exist
        print(f"Creating dataset folder...")
        for label in np.unique(y_train):
            os.makedirs(f"{self.data_dir}/labelled/{label}", exist_ok=True)
        os.makedirs(f"{self.data_dir}/unlabelled/", exist_ok=True)

        # Index for saving images
        idx = 0

        # Save training images to unlabelled/class folders
        print(f"Saving images...")
        for img, label in zip(x_train, y_train):
            np.save(f"{self.data_dir}/unlabelled/{label}-{idx}.npy", img)
            idx += 1

        # Save test images to labelled/class folders
        for img, label in zip(x_test, y_test):
            np.save(f"{self.data_dir}/labelled/{label}/{idx}.npy", img)
            idx += 1
    
    def build_dataset(self, subset):
        print(f"Building \"{subset}\" dataset...")

        xs, ys = [], []
        if subset == 'labelled':
            files = glob.glob(f"{self.data_dir}/{subset}/*/*.npy")
        elif subset == 'unlabelled':
            files = glob.glob(f"{self.data_dir}/{subset}/*.npy")
        else:
            raise ValueError(f"Invalid subset: {subset}")
        
        # Browse all files
        for f in files:
            arr = np.load(f)
            label = -1
            if subset == "labelled":
                label = int(f.split("/")[-2])  # Extract label from file path
            xs.append(arr)
            ys.append(label)

        # Shuffle xs and ys together
        xs, ys = np.array(xs), np.array(ys)
        indices = np.arange(len(xs))
        np.random.shuffle(indices)
        xs = xs[indices]
        ys = ys[indices]

        xs = np.stack(xs).astype("float32")[..., None] / 255.
        ys = np.array(ys, dtype="int64")

        print(f'\tTotal images: {len(xs)}')

        if subset == "labelled":
            return xs, ys    
        return xs, np.zeros_like(xs)
    
    def build(self):
        self.labelled = self.build_dataset("labelled")
        self.unlabelled = self.build_dataset("unlabelled")

    def make_iter(self, subset, shuffle=True):
        if subset == "labelled":
            xs, ys = self.labelled
        elif subset == "unlabelled":
            xs, ys = self.unlabelled
        else:
            raise ValueError(f"Invalid subset: {subset}")
        
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        if shuffle:
            ds = ds.shuffle(len(xs))
        return iter(ds.batch(self.batch_size).repeat())
    