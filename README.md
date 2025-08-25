# Sentiment Classification on IMDb Reviews

## Brief One Line Summary
Deep learning models (Neural Net, CNN, LSTM) applied on IMDb dataset of 50,000 reviews to classify sentiments with LSTM achieving the highest accuracy of 86.64%.

## Overview
Sentiment analysis is a common natural language processing (NLP) task that aims to classify text into positive or negative sentiments. In this project, we analyze IMDb movie reviews and build deep learning models to classify sentiments effectively. The focus is on comparing multiple architectures and selecting the best-performing model.

## Problem Statement
- Perform sentiment classification on IMDb movie reviews.  
- Handle raw unstructured text data through preprocessing and embeddings.  
- Compare different neural architectures to determine the most effective model.  

## Dataset
- Source: [IMDb Movie Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- Records: 50,000 reviews  
- Labels: Binary sentiment (positive = 1, negative = 0).  

## Tools and Technologies
- Python  
- Pandas, NumPy  
- TensorFlow / Keras  
- Scikit-learn  
- NLTK  

## Methods
- Text preprocessing (cleaning, tokenization, stopword removal).  
- Word embeddings using GloVe.  
- Models implemented:  
  - Simple Neural Network  
  - Convolutional Neural Network (CNN)  
  - Long Short-Term Memory (LSTM)  

## Key Insights
- Preprocessing and embeddings significantly improve classification performance.  
- CNNs capture local semantic features, but LSTMs outperform in long-sequence understanding.  
- Deep learning models generalize well compared to traditional ML baselines.  

## Dashboard / Model / Output
- Jupyter Notebook implementation with plots of training/validation accuracy and loss.  
- Confusion matrix showing classification performance.  
- Final LSTM model predictions on sample movie reviews.  

## How to Run this Project?
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/Sentiment-Classification.git
   cd Sentiment-Classification
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run the notebook
   ```bash
   jupyter notebook "b1_SentimentAnalysis_with_NeuralNetwork.ipynb"


## Results & Conclusion

Neural Network: Moderate performance.

CNN: Better at capturing local patterns in text.

LSTM: Achieved 86.64% accuracy, the best-performing model for sentiment classification.

Conclusion: LSTM networks are highly effective for text-based sentiment analysis, outperforming CNNs and simpler neural models in capturing long-term dependencies.

## Future Work

Experiment with Transformer-based architectures (BERT, DistilBERT).

Use data augmentation techniques to handle class imbalance and improve generalization.

Deploy the model as a web app or API for real-time sentiment analysis.


