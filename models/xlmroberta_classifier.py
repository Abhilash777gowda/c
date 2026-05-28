"""
XLM-RoBERTa fine-tuned multi-label classifier for CRIMSON-India.
Inherits all logic from TransformerClassifier with a different default model name.
"""

from models.transformer_classifier import TransformerClassifier
from utils.helpers import setup_logging

logger = setup_logging()


class XLMRoBertaClassifier(TransformerClassifier):
    """
    XLM-RoBERTa based classifier.
    Identical to TransformerClassifier but defaults to xlm-roberta-base
    and saves to models/saved_xlmroberta/.
    """

    def __init__(self, categories: list,
                 model_name: str = "xlm-roberta-base",
                 save_dir: str = "models/saved_xlmroberta",
                 max_length: int = 128):
        super().__init__(
            categories=categories,
            model_name=model_name,
            save_dir=save_dir,
            max_length=max_length,
        )
        logger.info(f"XLMRoBertaClassifier initialised (model={model_name}).")
