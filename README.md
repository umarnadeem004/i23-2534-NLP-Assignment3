# NLP Assignment 3: Transformers + RAG

## Overview

This project implements a three-stage NLP system for sentiment understanding and explanation generation:

1. **Part A: Encoder Model** - Multi-task encoder for sentiment classification and review length prediction
2. **Part B: Retrieval Module** - Similarity-based retrieval of relevant training examples
3. **Part C: Decoder Model** - Explanation generation using retrieved context (scaffolded)

## How to Run

1. Ensure Python 3.10+ is installed with PyTorch and required packages
2. Open `i232534-NLP-Assignment3.ipynb` in Jupyter
3. Run cells from top to bottom:
   - **Cells 1-5**: Dataset loading and preprocessing (creates train/val/test splits)
   - **Cells 6-9**: Encoder model architecture and training (Part A)
   - **Cells 10-11**: Test evaluation and embedding extraction for retrieval
   - **Cells 12-16**: Retrieval module and decoder setup (Parts B & C scaffolds)

## Results

### Part A: Encoder Performance (Test Set)
- **Sentiment Classification**: 82.59% accuracy
- **Length Bucket Prediction**: 99.96% accuracy
- Training completed in 10 epochs with early stopping
- Learning curves saved in `results/part_a_training_curves.png`

### Part B: Retrieval
- 25,200 training embeddings stored in `results/train_embeddings.npy`
- Cosine similarity-based retrieval with configurable top-k (default: 5)
- Tested on validation examples

## Directory Structure

```
.
├── i232534-NLP-Assignment3.ipynb    # Main notebook (run this)
├── README.md                         # This file
├── models/
│   ├── encoder_best.pt              # Best encoder checkpoint
│   └── encoder_final.pt             # Final encoder weights
├── results/
│   ├── train_embeddings.npy         # 25200 x 128 encoder embeddings
│   ├── train_sentiments.npy         # Corresponding sentiment labels
│   ├── train_indices.npy            # Index mapping
│   ├── train_reviews.csv            # Training set
│   ├── val_reviews.csv              # Validation set
│   ├── test_reviews.csv             # Test set
│   ├── amazon_reviews_subset.csv    # Full 36k sample subset
│   └── part_a_training_curves.png   # Training visualization
└── .git/                            # Git repository
```

## Dataset

- **Source**: Amazon Reviews dataset
- **Categories**: Beauty, Cellphones, Sports
- **Total Samples**: 36,000 (12,000 per category)
- **Split**: 70% train (25,200), 15% val (5,400), 15% test (5,400)
- **Preprocessing**: Text cleaning, tokenization, vocabulary construction from training data only

## Key Implementation Details

### Encoder
- **Architecture**: 2-layer Transformer encoder with 4 attention heads
- **Embedding Dimension**: 128
- **Sequence Length**: Fixed at 128 tokens
- **Vocabulary Size**: 19,913 tokens (min frequency: 2)
- **Loss**: Joint cross-entropy for sentiment + length prediction (weighted 1:0.5)

### Sentiment Labels
- 1-2 stars → Negative
- 3 stars → Neutral
- 4-5 stars → Positive

### Derived Feature
- Review length bucket: short (< 40 tokens), medium (40-100), long (>100)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- pandas, numpy, scikit-learn, matplotlib

## Notes

- All preprocessing and models implemented from scratch (no pretrained models)
- Attention mechanisms, multi-head attention, and encoder blocks are fully custom
- No use of `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder`
- Early stopping with patience=3 to prevent overfitting

## Next Steps

- Complete Part C: Decoder training and explanation generation
- Generate qualitative examples with commentary
- Perform ablation study (with/without retrieval)
- Compute perplexity metrics
- Write comprehensive report (3-5 pages)

---
**Due Date**: April 29, 2026
**Submission**: GitHub with incremental commits
