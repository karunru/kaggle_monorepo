"""exp018: BERT integration for enhanced feature extraction.

This experiment extends exp013 by integrating BERT attention mechanism
to replace global pooling and improve temporal feature extraction.

Based on:
- Kaggle notebook: https://www.kaggle.com/code/cody11null/public-bert-training-attempt
- Hugging Face Transformers BERT implementation

Key improvements:
- BERT self-attention mechanism for temporal feature extraction
- CLS token for effective feature aggregation
- Enhanced long-range dependency modeling
- Improved classification performance through better feature representation
"""
