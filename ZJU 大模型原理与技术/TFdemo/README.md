# Language Model Training

This project implements a language model using various architectures, including RNN and Transformer models. The goal is to train a model on a dataset derived from Wikipedia and evaluate its performance.

## Project Structure

```
language-model-training
├── data
│   ├── wiki.train.txt       # Training data for the language model
│   ├── wiki.valid.txt       # Validation data for the language model
│   └── wiki.test.txt        # Test data for the language model
├── src
│   ├── models
│   │   ├── __init__.py      # Marks the models directory as a Python package
│   │   ├── transformer.py    # Implementation of the Transformer model
│   │   └── rnn.py            # Implementation of the RNN model
│   ├── utils
│   │   ├── __init__.py      # Marks the utils directory as a Python package
│   │   ├── data_utils.py     # Utility functions for data processing and loading
│   │   └── train_utils.py    # Utility functions for training the model
│   ├── train.py              # Main script for training the language model
│   └── evaluate.py           # Evaluation logic for the trained model
├── config
│   └── default.json          # Default configuration settings
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd language-model-training
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place your training, validation, and test data in the `data` directory.

## Usage

To train the language model, run the following command:
```
python src/train.py
```

To evaluate the trained model, use:
```
python src/evaluate.py
```

## Configuration

The configuration settings, including hyperparameters and file paths, can be modified in `config/default.json`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.