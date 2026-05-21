import numpy as np


class BracketDataSource:
    def __init__(self, file_path, unmask_ids):
        self.sequences = np.load(file_path)
        self.unmask_ids = unmask_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
            "loss_mask": np.isin(seq[1:], self.unmask_ids).astype(np.float32)
        }
