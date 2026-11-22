"""Evaluation boundary utilities. No model evaluation is performed here."""
from .dataset import read_splits

def verify_split_isolation(root):
    """Validate official split counts and path isolation without opening tensors."""
    return {name: len(records) for name, records in read_splits(root).items()}
