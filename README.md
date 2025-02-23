# Intent Classification with CamemBERT

A French intent classification system using CamemBERT for natural language understanding. This repository includes a CLI for predictions/evaluation and a Jupyter notebook for model training.

## Features

- 🚀 Pre-trained CamemBERT model from Hugging Face
- 📊 Dual threshold system for handling uncertain predictions
- 📈 Training notebook with early stopping
- 🖥️ CLI interface with three modes:
  - Single text prediction
  - CSV file evaluation
  - Interactive mode
- 🧹 Text preprocessing with stopword removal and punctuation cleaning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intent-classifier.git
   cd intent-classifier
   ```

2. Install dependencies:
   ```bash
   python install_dependencies.py
   ```


## Usage

### Command Line Interface

```bash
# Single prediction
python cli.py --text "Où est ma valise perdue ?"

# Evaluate CSV file
python cli.py --csv test_data.csv --threshold1 0.3 --threshold2 0.85

# Interactive mode
python cli.py
```

### Key Arguments

| Argument     | Description                              | Default |
|-------------|----------------------------------|---------|
| --text      | Text to classify                     | -       |
| --csv       | CSV file for evaluation              | -       |
| --threshold1 | Confidence threshold for OOS         | 0.2     |
| --threshold2 | Special threshold for lost luggage  | 0.8     |

## Training

Use the `nn_classifier.ipynb` notebook.

- Requires dataset at `/content/intent-detection-extended-v2.csv`
- Implements:
  - Stratified data splitting
  - 200+ epoch training with early stopping
  - Model serialization

## Model Performance

**Accuracy:** 0.8929

| Intent            | Precision | Recall | F1-score | Support |
|-------------------|-----------|--------|----------|---------|
| book_flight      | 1.00      | 1.00   | 1.00     | 2       |
| book_hotel       | 0.75      | 1.00   | 0.86     | 3       |
| carry_on         | 1.00      | 1.00   | 1.00     | 3       |
| flight_status    | 1.00      | 1.00   | 1.00     | 3       |
| lost_luggage     | 1.00      | 1.00   | 1.00     | 3       |
| out_of_scope     | 0.80      | 0.67   | 0.73     | 6       |
| translate        | 1.00      | 1.00   | 1.00     | 3       |
| travel_alert     | 1.00      | 1.00   | 1.00     | 2       |
| travel_suggestion | 0.67     | 0.67   | 0.67     | 3       |

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- Pandas
- scikit-learn
- NLTK

## Additional Recommendations

To improve usability, consider adding:
1. A `requirements.txt` file
2. Sample test data
3. A `LICENSE` file
4. Documentation for model hosting on Hugging Face Hub


