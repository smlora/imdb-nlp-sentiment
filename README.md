# IMDB Sentiment Analysis & Text Generation with LSTM and GPT-2

This project explores natural language processing (NLP) through two deep learning tasks: sentiment classification and generative text modeling. Using the IMDB movie reviews dataset, the project implements a Bidirectional LSTM for binary sentiment analysis and fine-tunes a pretrained GPT-2 model to generate positive reviews.

## Objective

- Build and train a Bidirectional LSTM model to classify movie reviews as positive or negative.
- Fine-tune a GPT-2 model on the IMDB dataset to generate new, coherent movie reviews with a positive sentiment.
- Evaluate generated reviews using the trained sentiment classifier.

## Dataset

- **Source**: [IMDB Movie Reviews](https://keras.io/api/datasets/imdb/)
- **Size**: 50,000 labeled reviews (25k train / 25k test)
- **Labels**: 0 = Negative, 1 = Positive

## Model Overview

### LSTM Sentiment Classifier:
- Embedding layer → Bidirectional LSTM → Dense output
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

### GPT-2 Fine-Tuning:
- Pretrained GPT-2 model fine-tuned on positive IMDB reviews
- Uses text generation prompts and temperature sampling
- Texts evaluated by the LSTM sentiment classifier

## Results

- The Bidirectional LSTM achieved strong binary classification accuracy on IMDB test data.
- The fine-tuned GPT-2 model successfully generated original, positively biased movie reviews.
- Classifier predictions confirmed that generated reviews were mostly rated as positive.

## Project Structure

```
imdb-nlp-sentiment/
├── notebooks/
│   └── Steven_Lora_MSIT675_Project3.ipynb
├── README.md
└── requirements.txt
```

## How to Run

> This project is implemented in a Jupyter Notebook environment.

1. Clone the repo and navigate to the notebooks folder:
   ```
   git clone https://github.com/smlora/imdb-nlp-sentiment.git
   cd imdb-nlp-sentiment/notebooks
   ```

2. Install dependencies:
   ```
   pip install -r ../requirements.txt
   ```

3. Open the notebook:
   ```
   jupyter notebook Steven_Lora_MSIT675_Project3.ipynb
   ```

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

## Technologies Used

- Python
- TensorFlow / Keras
- GPT-2 (via Keras Hub or Hugging Face)
- NumPy
- Matplotlib
- Jupyter Notebook

## Contact

Created by [Steven Lora](https://www.linkedin.com/in/smlora/)
