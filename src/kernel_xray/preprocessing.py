"""Public training-only preprocessing API; implementation shared with Dataset."""
from .dataset import NormalizationStats, fit_training_statistics
__all__ = ['NormalizationStats', 'fit_training_statistics']
