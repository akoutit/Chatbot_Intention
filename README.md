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
   git clone https://github.com/akoutit/Chatbot_Intention.git
   cd Chatbot_Intention
   ```

2. Install dependencies:
   ```bash
   python install_dependencies.py
   ```


## Usage

### Command Line Interface

```bash
# Interactive mode
python cli.py

# Evaluate CSV file
python cli.py --csv test_data.csv --threshold1 0.3 --threshold2 0.85
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

- Requires dataset at `/content/intent-detection-augmented.csv`
- Implements:
  - Stratified data splitting
  - 200+ epoch training with early stopping
  - Model serialization

## Model Performance

**Accuracy:** 0.9623  

| Intent            | Precision | Recall | F1-score | Support |
|-------------------|-----------|--------|----------|---------|
| book_flight      | 1.00      | 1.00   | 1.00     | 5       |
| book_hotel       | 1.00      | 1.00   | 1.00     | 5       |
| carry_on         | 1.00      | 1.00   | 1.00     | 6       |
| flight_status    | 1.00      | 1.00   | 1.00     | 5       |
| lost_luggage     | 0.83      | 1.00   | 0.91     | 5       |
| out_of_scope     | 1.00      | 0.80   | 0.89     | 10      |
| translate        | 1.00      | 1.00   | 1.00     | 6       |
| travel_alert     | 0.83      | 1.00   | 0.91     | 5       |
| travel_suggestion | 1.00     | 1.00   | 1.00     | 6       |

### Global Metrics  

| Metric           | Value  |
|-----------------|--------|
| **Accuracy**        | **0.9623** |
| **Macro Avg Precision** | 0.96  |
| **Macro Avg Recall** | 0.98  |
| **Macro Avg F1-score** | 0.97  |
| **Weighted Avg Precision** | 0.97 |
| **Weighted Avg Recall** | 0.96 |
| **Weighted Avg F1-score** | 0.96 |

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- Pandas
- scikit-learn
- NLTK



