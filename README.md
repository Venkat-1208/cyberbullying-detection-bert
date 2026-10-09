# Cyberbullying Detection Using BERT and Machine Learning

## Overview
This project explores binary text classification for cyberbullying detection using tweet text, pretrained BERT embeddings, and Logistic Regression.

## Dataset
- File: `CyberBullying_tweets.csv`
- Records: 11,100
- Columns: `Text`, `CB_Label`
- Target distribution: 5,550 samples per label

The meaning of each label should be confirmed from the original dataset documentation.

## Methodology
1. Load and inspect the dataset.
2. Clean text by lowercasing and removing URLs, mentions, and special characters.
3. Tokenize the cleaned text.
4. Extract contextual embeddings using pretrained `bert-base-uncased`.
5. Train a Logistic Regression classifier.
6. Evaluate using accuracy and a classification report.

## Recorded Results
The existing notebooks contain recorded accuracy results ranging from 75.63% to 76.98%, depending on the experiment.

These are previously saved notebook results and have not yet been independently reproduced.

## Technologies
- Python
- Jupyter Notebook
- Pandas and NumPy
- NLTK
- PyTorch
- Hugging Face Transformers
- Scikit-learn

## Project Files
- `cyberbullyingML.ipynb` — Main notebook candidate
- `cyberbullyingML (1).ipynb` — Additional experiment
- `cyberbullyingML (2).ipynb` — Additional experiment
- `CyberBullying_tweets.csv` — Dataset
- `README.md` — Project documentation
- `.gitignore` — Files excluded from Git

## Running the Project
1. Install Python and Jupyter Notebook.
2. Install the required Python libraries.
3. Place the dataset in the project directory.
4. Open the selected notebook.
5. Run the cells in order.

The pretrained BERT model must be downloaded when it is not already cached locally.

## Limitations
The current implementation uses frozen pretrained BERT embeddings with Logistic Regression. Performance on this dataset does not guarantee reliable detection across different platforms, languages, or communities.

## Future Improvements
- Add a confusion matrix and detailed evaluation metrics.
- Optimize batch embedding extraction.
- Improve reproducibility and dependency management.
- Compare additional classification approaches.
- Develop an interface for testing new text.

## Responsible Use
Predictions are model estimates, not definitive judgments. Human review is recommended before taking action based on a prediction.
