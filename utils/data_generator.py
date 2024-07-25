from tensorflow.keras.utils import Sequence
import numpy as np

class MyDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_file_paths = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = []
        for file_path in batch_file_paths:
            X_batch.append(np.load(file_path))

        X_batch = np.array(X_batch)
        return X_batch, np.array(batch_labels)

