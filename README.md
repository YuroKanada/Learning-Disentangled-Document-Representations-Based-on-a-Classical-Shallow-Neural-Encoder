# Disentangled Document Representations with Shallow Neural Encoders

This repository contains the official implementation of our iiWAS 2025 paper:

**"Learning Disentangled Document Representations Based on a Classical Shallow Neural Encoder"**  
Yuro Kanada, Yoshiyuki Shoji, Sumio Fujita

## Objective

This project proposes a document embedding method that produces **disentangled representations**, where each embedding dimension:

1. Reflects **real-world relational distances** in the latent space,  
2. Is **statistically independent** of other dimensions, and  
3. Carries a **semantically interpretable meaning**.

The method extends a classic `doc2vec`-style shallow neural encoder by introducing:

- An **auxiliary metadata prediction task** (guidance task), and  
- A **KL divergence-based regularization term** that encourages embeddings to follow a multivariate standard normal distribution.

## Method Overview

[Detailed_overview_e.pdf](https://github.com/user-attachments/files/21370597/Detailed_overview_e.pdf)

During training, the model takes as input:

- A document ID and  
- A word within the document (one-hot encoded)

and jointly learns to:

- Predict surrounding context words (as in doc2vec), and  
- Predict associated metadata (e.g., genre or sentiment)

In parallel, the distribution of document embeddings is regularized toward a multivariate standard normal distribution via closed-form KL divergence, promoting **dimension-wise independence**.

## Architecture Summary

- **Base encoder**: PV-DM + PV-DBOW (doc2vec-style)
- **Learning tasks**:
  - Context word prediction
  - Metadata classification
- **Regularization**:
  - KL divergence with standard Gaussian distribution

## Datasets

- **Synthetic dataset**: 10,000 short documents assigned to 20 latent topics.  
- **IMDb review dataset**: 50,000 reviews for 1,000 movies (with manually annotated genres).

## Repository Structure

- **common/**
  - `np.py` – unified NumPy/CuPy interface  

- **config/**
  - `config.py` – global configuration (e.g., GPU flag)  

- **models/**
  - `__init__.py`
  - `doc2dr.py` – main Doc2DR model implementation  
  - `embedding.py` – embedding layers  
  - `loss.py` – loss functions (NegativeSamplingLoss, disentangled_loss, etc.)  
  - `scheduler.py` – SigmoidAnnealingScheduler  

- **scripts/**
  - `train_doc2dr.py` – training entry script  

- **trainers/**
  - `__init__.py`
  - `trainer.py` – training loop and utilities  

- **utils/**
  - `__init__.py`
  - `data.py` – data preprocessing (create contexts/targets, etc.)  
  - `math_utils.py` – mathematical utilities (KL divergence, etc.)  
  - `optimizers.py` – Adam optimizer  
  - `sampler.py` – unigram and genre samplers  

- **traindata_reviewWord500_param.pkl** – example preprocessed dataset (pickle)
