# Emotion Classification using Tweets

This project focuses on classifying emotions in Twitter posts using deep learning models. It explores the effectiveness of a graph-based preprocessing technique to enhance model performance by capturing word relationships and multi‑word expressions.

## Overview

The goal is to accurately classify tweets into six emotion categories: sadness, joy, love, anger, fear, and surprise. Three baseline models are evaluated:

- **RNN** (Recurrent Neural Network)
- **LSTM** (Long Short‑Term Memory)
- **DistilBERT** (a distilled version of BERT)

Each model is trained on both raw tweet texts and on texts processed with a graph‑based approach that builds co‑occurrence graphs and merges consecutive words based on edge weights.

## Dataset

The dataset is the [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion) from Hugging Face (`dair-ai/emotion`). It contains tweets labeled with one of the six emotions.

- **Split configuration**: 20,000 examples (train/validation)
- **Unsplit configuration**: 416,809 examples (single train split)

To ensure balanced classes, we used the unsplit configuration and created our own balanced training, validation, and test sets (60,000 / 6,000 / 18,000 samples respectively).

## Methods

### Graph‑Based Preprocessing

We implemented a custom graph‑based preprocessing pipeline (see [graph](https://github.com/LyzJames/graph)) that:

1. Builds a co‑occurrence graph to capture word relationships.
2. Merges consecutive words based on edge weights to form multi‑word expressions.
3. Removes stopwords.

This processing is applied to the training, validation, and test sets before feeding them into the models.

### Models

- **RNN** – A simple recurrent neural network with an embedding layer, a SimpleRNN layer, and two dense layers.
- **LSTM** – Similar architecture but with an LSTM layer to better capture long‑range dependencies.
- **DistilBERT** – A pre‑trained transformer model fine‑tuned for sequence classification.

All models were implemented in TensorFlow/Keras (for RNN/LSTM) and Transformers (for DistilBERT). Training was performed on Google Colab with GPU acceleration.

### Evaluation

Models were evaluated on the balanced test set using accuracy, F1 score, precision, and recall. For each model, we compared performance on raw data versus graph‑processed data.

## Results

| Model | Accuracy (raw) | Accuracy (graph) | F1 Score (raw) | F1 Score (graph) |
|-------|----------------|------------------|----------------|------------------|
| RNN   | 0.8275         | 0.8869           | 0.8236         | 0.8862           |
| LSTM  | 0.9327         | 0.9277           | 0.9321         | 0.9272           |
| DistilBERT | 0.9453    | 0.9471           | 0.9450         | 0.9469           |

- **RNN** showed a significant improvement with graph‑based preprocessing (+6% accuracy).
- **LSTM** and **DistilBERT** performed similarly on both raw and graph‑processed data, with minor differences.

## Discussion

The improvement in RNN performance is attributed to its ability to better capture sequential dependencies when word relationships are highlighted by the graph preprocessing. LSTM already handles long‑range dependencies well, making the additional graph structure less impactful. DistilBERT’s pre‑trained contextual embeddings may be robust enough that the graph‑based features add little extra benefit.

### Strengths
- Graph preprocessing effectively enhances simple sequential models like RNN.
- Balanced dataset ensures fair evaluation across all emotion classes.

### Weaknesses
- The approach did not improve more advanced models (LSTM, DistilBERT).
- Only three models were tested; other architectures might behave differently.

### Future Work
- Explore graph‑based preprocessing with other transformer models (GPT, T5, RoBERTa).
- Optimize graph construction parameters (e.g., edge weight thresholds, window sizes) to better suit each model.
- Combine graph features with attention mechanisms for hybrid models.

## References

1. Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). *CARER: Contextualized Affect Representations for Emotion Recognition*. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 3687–3697). Association for Computational Linguistics.  
   [https://doi.org/10.18653/v1/D18-1404](https://doi.org/10.18653/v1/D18-1404)

2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108.  
   [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)

3. Abdul-Mageed, M., & Ungar, L. (2017). *EmoNet: Fine-Grained Emotion Detection with Gated Recurrent Neural Networks*. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics* (Volume 1: Long Papers), pp. 718–728.  
   [https://aclanthology.org/P17-1067/](https://aclanthology.org/P17-1067/)

4. Bianchi, F., Nozza, D., & Hovy, D. (2021). *FEEL-IT: Emotion and Sentiment Classification for the Italian Language*. In *Proceedings of the Eleventh Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis*, pp. 76–83.  
   [https://aclanthology.org/2021.wassa-1.8/](https://aclanthology.org/2021.wassa-1.8/)

## Requirements

- Python 3.9+
- TensorFlow 2.x
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- scikit‑learn
- pandas
- matplotlib
- nltk (for stopwords)
- Google Colab (recommended for GPU)

The notebook expects pre‑trained model files (`RNN_model.keras`, `LSTM_model.keras`, `DistilBERT_model`, etc.) to be present in Google Drive. The graph‑based preprocessing code is available in a separate GitHub repository ([https://github.com/LyzJames/graph](https://github.com/LyzJames/graph)), which is cloned during execution.