import torch
import pandas as pd
import numpy as np
import string
import argparse
import joblib
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

# Charger le modèle depuis Hugging Face
MODEL_NAME = "Koutit/intention_classifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CamembertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)

# Charger le label encoder
LABEL_ENCODER_PATH = "Models/label_encoder.pkl"
label_encoder = joblib.load(LABEL_ENCODER_PATH)

class IntentClassifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.stop_words = set(stopwords.words("french"))

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join([word for word in text.split() if word not in self.stop_words])
        return text

    def predict_with_threshold(self, text: str) -> str:
        text = self.preprocess_text(text)
        encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        max_prob = np.max(probs)
        predicted_label = np.argmax(probs)

        if max_prob < self.threshold:
            return "out_of_scope"

        return label_encoder.inverse_transform([predicted_label])[0]

    def evaluate(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df["predicted_label"] = df["text"].apply(self.predict_with_threshold)
        accuracy = accuracy_score(df["label"], df["predicted_label"])
        print("Précision du modèle:", accuracy)
        print(classification_report(df["label"], df["predicted_label"]))

# Interface CLI
def main():
    parser = argparse.ArgumentParser(description="Utiliser le modèle de classification d'intentions.")
    parser.add_argument("--text", type=str, help="Le texte à classifier")
    parser.add_argument("--csv", type=str, help="Fichier CSV pour évaluer le modèle")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de confiance pour la classification")
    args = parser.parse_args()

    classifier = IntentClassifier(threshold=args.threshold)

    if args.csv:
        classifier.evaluate(args.csv)
    elif args.text:
        prediction = classifier.predict_with_threshold(args.text)
        print(f"Prédiction: {prediction}")
    else:
        print("Mode interactif. Tapez 'exit' pour quitter.")
        while True:
            text = input("Entrez une phrase: ")
            if text.lower() == "exit":
                break
            print("Prédiction:", classifier.predict_with_threshold(text))

if __name__ == "__main__":
    main()
