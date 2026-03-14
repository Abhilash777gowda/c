import pandas as pd
import numpy as np
import logging
import os
import torch
from torch.utils.data import Dataset

def setup_logging(log_filename="pipeline.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class NewsDataset(Dataset):
    """
    Standard PyTorch Dataset for Transformer models.
    Expects input encodings (from tokenizer) and numeric labels.
    """
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
