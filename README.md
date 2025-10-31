# 💬 Deep Sequential Modeling: Comparative Sentiment Analysis with RNN, LSTM, and Attention

## 🌟 Project Overview

This project implements and compares various **Recurrent Neural Network (RNN)** architectures for a binary **Sentiment Analysis** task. The goal is to classify text input (e.g., movie reviews, social media data) as either **positive or negative**.

The work features an in-depth study of:
1.  **Simple RNN** (Baseline)
2.  **Long Short-Term Memory (LSTM)**
3.  **Bidirectional LSTM (Bi-LSTM)**

A critical analysis is made concerning the **trade-off between model performance (accuracy) and training complexity/speed**, laying the groundwork for selecting production-ready NLP models. This project was developed as part of advanced coursework in Deep Learning.

***

## ✨ Core Technical Highlights

| Category | Methodology / Concept | Feature Demonstrated |
| :--- | :--- | :--- |
| **Deep Learning Framework** | **Keras / TensorFlow** | Hands-on experience in building, compiling, and training sequential neural networks. |
| **Architectures** | **LSTM, Bi-LSTM** | Mastery in leveraging advanced recurrent units to capture long-range dependencies and bidirectional context within sequences. |
| **Data Preparation** | **Tokenization & Embedding Layers** | Standard NLP workflow application, converting raw text into numerical vector representations. |
| **Comparative Analysis** | **Performance vs. Time/Complexity** | Critical evaluation of architectural design choices based on accuracy gains versus computational overhead. |
| **Advanced Proposal** | **Attention Mechanism** | Identification of architectural limitations in LSTMs and proposition of an Attention layer for enhanced focus and interpretability. |

***

## 🧠 Architectures and Methodology

### 1. General Model Structure

All comparative models share a fundamental deep learning pipeline:

$$\text{Input Sequence} \rightarrow \text{Embedding Layer} \rightarrow \text{Recurrent Layer(s)} \rightarrow \text{Dense Layers} \rightarrow \text{Output (Binary Sentiment)}$$

### 2. Comparison Summary

The analysis highlighted the expected trade-offs (as confirmed in the original notebook code/structure):

* **Simple RNN:** Fastest model, but struggles with vanishing gradients, leading to the lowest performance. Serves as a performance baseline.
* **LSTM:** Provides a significant boost in performance over the simple RNN by managing long-term memory, striking a good balance between speed and accuracy.
* **Bi-LSTM:** Achieves the **highest overall accuracy** by processing the sequence both forwards and backward, maximizing contextual understanding.

## 💡 Future Work & R&D Insights

The next logical step for this project—especially relevant for R&D—is migrating from purely recurrent architectures to models capable of identifying **salient information** more efficiently.

* **Proposal:** Integrate a **Self-Attention or Multi-Head Attention** layer into the model pipeline.
* **Advantage:** The Attention mechanism would allow the model to **selectively focus on critical sentiment-bearing words** (e.g., "awful," "brilliant," "not bad") regardless of their position in the sequence, improving both performance and model interpretability.

***

## 🚀 Reproduction

### Files
* **`703_RNN_sentiment_analysis_problem.ipynb`:** The complete Jupyter Notebook containing all model implementations, training loops, and comparative results.

### Prerequisites (Assumed from Notebook)
* Python 3.x
* **TensorFlow**
* **Keras**
* NumPy
* Pandas

### Execution Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/xavierplt/](https://github.com/xavierplt/)[Sentiment-analysis].git
    cd [REPO_NAME]
    ```
2.  **Install Dependencies:**
    ```bash
    # Install the core required libraries
    pip install tensorflow numpy pandas jupyter
    ```
3.  **Run the Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open and run the cells in **`703_RNN_sentiment_analysis_problem.ipynb`** to reproduce the training and comparative plots.
