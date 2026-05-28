"""
MuRIL fine-tuned multi-label classifier for CRIMSON-India.
Targets Hindi and other Indian-language articles.
Inherits all logic from TransformerClassifier with google/muril-base-cased as default.
"""

from models.transformer_classifier import TransformerClassifier
from utils.helpers import setup_logging

logger = setup_logging()


class MuRILClassifier(TransformerClassifier):
    """
    MuRIL (Multilingual Representations for Indian Languages) based classifier.
    Identical to TransformerClassifier but defaults to google/muril-base-cased
    and saves to models/saved_muril/.
    """

    def __init__(self, categories: list,
                 model_name: str = "google/muril-base-cased",
                 save_dir: str = "models/saved_muril",
                 max_length: int = 128):
        super().__init__(
            categories=categories,
            model_name=model_name,
            save_dir=save_dir,
            max_length=max_length,
        )
        logger.info(f"MuRILClassifier initialised (model={model_name}).")
