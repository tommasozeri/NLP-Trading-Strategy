# Financial Sentiment Analysis: From Tweet Classification to Trading Signals

**Author:** Tommaso Zeri  
**Context:** Advanced Machine Learning, MSc in Quantitative Finance, Alma Mater Studiorum, Università di Bologna  
**Date:** May 2026

---

## 📌 Project Overview

This repository contains an end-to-end Natural Language Processing (NLP) pipeline designed to transform financial tweets into actionable trading signals for three major U.S. equities: AAPL, TSLA, and MSFT.

The core research question motivating this work is: **Does a more accurate NLP classifier for financial sentiment always produce a more profitable trading signal?** The project explores whether trading alpha is strictly derived from model accuracy or if it is inherently tied to an asset's unique social media ecosystem.

---

## 📊 Datasets

The pipeline leverages two distinct text corpora:

- **Primary Training Corpus:** The HuggingFace `zeroshot/twitter-financial-news-sentiment` dataset  
  Contains 11,931 financial tweets annotated by domain experts into three classes: Bearish, Neutral, and Bullish.

- **Real-Timestamp Backtest Corpus:** Sourced from Kaggle  
  Contains 80,793 tweets spanning October 2021 to September 2022. Features real UTC timestamps, which are critical for causally aligning social media sentiment with subsequent price movements.

---

## 🧠 Methodology & Models

Three model families of increasing architectural complexity were implemented and compared:

### Baseline (Classical Machine Learning)
TF-IDF vectorization paired with classical classifiers: Logistic Regression, LinearSVC, and SVM (RBF).

### Deep Learning
Bidirectional LSTM (BiLSTM) network initialized with domain-agnostic pre-trained GloVe embeddings.

### Transformers
FinBERT model (BERT pre-trained on 1.8-billion-word financial corpus) fine-tuned using a two-phase strategy to prevent catastrophic forgetting.

### Preprocessing Strategy
- **Baseline & BiLSTM:** Aggressive normalization
- **FinBERT:** Minimal normalization (case-sensitive, preserves punctuation as syntactic cue)

---

## 💡 Key Findings

### 1. Classification Performance

- **Domain-specific pre-training dominates architectural complexity**
- **FinBERT:** F1 Macro = 0.841 (best performance)
- **BiLSTM:** F1 = 0.760 (marginal improvement over baseline)
- **LinearSVC (Baseline):** F1 = 0.754
- FinBERT's most significant contribution: +19.4% improvement in identifying the Bearish class (hardest due to financial negation and domain jargon)

### 2. Trading Signal Effectiveness

**Key Insight:** A more accurate classifier does not always produce a better trading signal; effectiveness is a joint property of the classifier and the asset's social-media ecosystem.

#### TSLA (Sentiment-Driven Regime)
- Highly predictive social media sentiment
- Confidence-weighted FinBERT signal: **+19.91% cumulative return** (Sharpe 0.605)
- Outperformed buy-and-hold (+3.79%)

#### AAPL (Non-Predictive Regime)
- Social media sentiment carried no directional information
- Signals underperformed buy-and-hold strategy

#### MSFT (Contrarian Regime)
- Retail sentiment acted as a contrarian indicator
- Investors expressed bullish sentiment near local price maxima (bearish near minima)
- Resulted in negative correlation with forward returns

---

## ⚙️ Environment & Setup

The project was executed in **Google Colab Pro** utilizing an **A100 GPU**.
