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
# Tapez 'exit' pour quitter.
python cli.py

# Evaluate CSV file
python cli.py --csv test_data.csv

# Evaluate CSV file choosing other thresholds 
python cli.py --csv test_data.csv --threshold1 0.3 --threshold2 0.85
```

💡 **Note** : Lors de la première exécution, la commande CLI peut prendre quelques secondes pour télécharger et installer le modèle hébergé sur Hugging Face. Une fois cette étape terminée, les exécutions suivantes seront fluides et rapides. 🚀  


### Key Arguments

| Argument     | Description                              | Default |
|-------------|----------------------------------|---------|
| --text      | Text to classify                     | -       |
| --csv       | CSV file for evaluation              | -       |
| --threshold1 | Confidence threshold for OOS         | 0.2     |
| --threshold2 | Special threshold for lost luggage  | 0.8     |

## Training

Use the `NN_classifier.ipynb` notebook.

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



## Training & Benchmark

### Training 

For training, we used data augmentation via chatbots, generating diverse utterances to enrich the dataset.  
The training was conducted using the `NN_classifier.ipynb` and `SVM_classifier.ipynb` notebook with the datasets located at `/content/intent-detection-augmented.csv` and `/content/intent-detection-Train.csv`.  
The notebook implements:
- **Stratified data splitting** to ensure class balance.
- **200+ epoch training** with **early stopping**.
- **Model serialization** for reuse.

We benchmarked two models:
1. **SVM** using **CamemBERT embeddings**.
2. **Neural Network** using the **CamemBERTClassifier pretrained model**.

To evaluate the generalization of each classifier, we created a **custom minimal dataset (`Data/intent-detection-minimal.csv`)**, where we manually added verbatims for each label.

| Text                                                     | Label            |
|-----------------------------------------------------------|------------------|
| Comment dit-on 'bonjour' en espagnol ?                    | translate        |
| Peux-tu traduire cette phrase en allemand ?               | translate        |
| Y a-t-il des restrictions de voyage pour le Brésil ?      | travel_alert     |
| Mon pays de destination est-il sous alerte rouge ?        | travel_alert     |
| Peux-tu vérifier si mon vol AF456 est à l'heure ?         | flight_status    |
| Est-ce que le vol pour Madrid est retardé ?               | flight_status    |
| Ma valise a disparu à l'aéroport, comment la retrouver ?  | lost_luggage     |
| J'ai perdu mes bagages, que dois-je faire ?               | lost_luggage     |
| As-tu une idée de voyage pour un week-end en Europe ?     | travel_suggestion|
| Quelle est la meilleure destination pour un voyage solo ? | travel_suggestion|
| Puis-je prendre un sac à dos en cabine ?                  | carry_on         |
| Quelles sont les dimensions autorisées pour un bagage cabine ? | carry_on   |
| Je voudrais réserver un hôtel à Rome.                     | book_hotel       |
| Peux-tu m’aider à trouver un hébergement à Tokyo ?        | book_hotel       |
| Quels sont les vols disponibles pour Londres cette semaine ? | book_flight  |
| Je veux réserver un billet d’avion pour New York.         | book_flight      |
| Peux-tu me donner la météo pour demain ?                  | out_of_scope     |
| Quel est le score du dernier match de football ?          | out_of_scope     |

### Benchmark Results

We tested four configurations:

#### **1. SVM trained on the original dataset**
```bash
python .\svm_cli.py --csv .\Data\intent-detection-minimal.csv
```
**Accuracy:** 0.1667  

| Intent             | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| book_flight       | 0.00      | 0.00   | 0.00     | 2       |
| book_hotel        | 0.00      | 0.00   | 0.00     | 2       |
| carry_on          | 0.00      | 0.00   | 0.00     | 2       |
| flight_status     | 0.00      | 0.00   | 0.00     | 2       |
| lost_luggage      | 0.00      | 0.00   | 0.00     | 2       |
| out_of_scope      | 0.12      | 1.00   | 0.21     | 2       |
| translate         | 1.00      | 0.50   | 0.67     | 2       |
| travel_alert      | 0.00      | 0.00   | 0.00     | 2       |
| travel_suggestion | 0.00      | 0.00   | 0.00     | 2       |

#### **2. SVM trained on the augmented dataset**
```bash
python .\svm_cli.py --csv .\Data\intent-detection-minimal.csv
```
**Accuracy:** 0.8889  

| Intent             | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| book_flight       | 0.67      | 1.00   | 0.80     | 2       |
| book_hotel        | 1.00      | 1.00   | 1.00     | 2       |
| carry_on          | 1.00      | 1.00   | 1.00     | 2       |
| flight_status     | 1.00      | 0.50   | 0.67     | 2       |
| lost_luggage      | 1.00      | 1.00   | 1.00     | 2       |
| out_of_scope      | 0.67      | 1.00   | 0.80     | 2       |
| translate         | 1.00      | 1.00   | 1.00     | 2       |
| travel_alert      | 1.00      | 1.00   | 1.00     | 2       |
| travel_suggestion | 1.00      | 0.50   | 0.67     | 2       |

#### **3. CamemBERTClassifier trained on the original dataset**
```bash
python .\cli.py --csv .\Data\intent-detection-minimal.csv
```
**Accuracy:** 0.8889  

| Intent             | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| book_flight       | 0.67      | 1.00   | 0.80     | 2       |
| book_hotel        | 1.00      | 1.00   | 1.00     | 2       |
| carry_on          | 1.00      | 1.00   | 1.00     | 2       |
| flight_status     | 1.00      | 0.50   | 0.67     | 2       |
| lost_luggage      | 1.00      | 1.00   | 1.00     | 2       |
| out_of_scope      | 1.00      | 1.00   | 1.00     | 2       |
| translate         | 1.00      | 1.00   | 1.00     | 2       |
| travel_alert      | 0.67      | 1.00   | 0.80     | 2       |
| travel_suggestion | 1.00      | 0.50   | 0.67     | 2       |

#### **4. CamemBERTClassifier trained on the augmented dataset**
```bash
python .\cli.py --csv .\Data\intent-detection-minimal.csv
```
**Accuracy:** 1.0000  

| Intent             | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| book_flight       | 1.00      | 1.00   | 1.00     | 2       |
| book_hotel        | 1.00      | 1.00   | 1.00     | 2       |
| carry_on          | 1.00      | 1.00   | 1.00     | 2       |
| flight_status     | 1.00      | 1.00   | 1.00     | 2       |
| lost_luggage      | 1.00      | 1.00   | 1.00     | 2       |
| out_of_scope      | 1.00      | 1.00   | 1.00     | 2       |
| translate         | 1.00      | 1.00   | 1.00     | 2       |
| travel_alert      | 1.00      | 1.00   | 1.00     | 2       |
| travel_suggestion | 1.00      | 1.00   | 1.00     | 2       |

### **Conclusion**
- The **SVM** model struggles to generalize when trained on the original dataset, but data augmentation significantly improves its performance.
- The **CamemBERTClassifier** performs well even with the original dataset and achieves **perfect accuracy** when trained with data augmentation.
- **Data augmentation via chatbots significantly improves generalization**, as demonstrated by the **large performance gap between models trained with and without augmentation**.
- **CamemBERTClassifier outperforms SVM** on this task, highlighting the power of transformer-based models in intent classification.






