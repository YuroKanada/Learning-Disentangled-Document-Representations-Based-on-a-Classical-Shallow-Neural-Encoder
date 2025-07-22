This repository contains the implementation of the method proposed in our iiWAS 2025 paper:
**"Learning Disentangled Document Representations Based on a Classical Shallow Neural Encoder"**  
by Yuro Kanada, Yoshiyuki Shoji, and Sumio Fujita.

## Overview

This method aims to learn document embeddings where:
- Each dimension is **statistically independent**,
- Each dimension captures a **semantically interpretable** feature.

To achieve this, we enhance a classic `doc2vec` encoder with:
1. **A guidance task** that predicts document metadata (e.g., genre),
2. **KL divergence regularization** that encourages the embedding distribution to approximate a multivariate standard normal distribution.

##  Model Architecture

- Base: Shallow neural network (PV-DM + PV-DBOW) with Negative Sampling.
- Dual objectives:
  - Context word prediction
  - Metadata classification
- Regularizer: KL divergence between empirical embedding distribution and standard multivariate Gaussian.

## Repository Structure

