import torch
import pandas as pd
import numpy as np
import string
import argparse
import joblib
import nltk
from typing import List
from nltk.corpus import stopwords
from transformers import CamembertTokenizer, CamembertModel
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)



# Classe pour la classification des intentions
class IntentClassifier:
    def __init__(self, model_path="Models/svm_model_augmented.pkl", label_encoder_path="Models/label_encoder.pkl", threshold=0.3):
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.model = CamembertModel.from_pretrained("camembert-base")
        self.label_encoder = joblib.load(label_encoder_path)
        self.stop_words = set(stopwords.words("french"))
        self.threshold = threshold  # Seuil de confiance
        
        # Charger uniquement le modèle `.pkl`
        self.classifier = joblib.load(model_path)

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join([word for word in text.split() if word not in self.stop_words])
        return text

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def predict_with_threshold_pkl(self, text: str) -> str:
        """Prédiction avec gestion de la classe inconnue."""
        text = self.preprocess_text(text)
        embedding = self.get_embedding(text).reshape(1, -1)

        probs = self.classifier.predict_proba(embedding)
        max_prob = np.max(probs)
        predicted_label = np.argmax(probs)

        if max_prob < self.threshold:
            return "out_of_scope"  # Classe inconnue

        return self.label_encoder.inverse_transform([predicted_label])[0]

    def evaluate(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df["predicted_label"] = df["text"].apply(self.predict_with_threshold_pkl)
        accuracy = accuracy_score(df["label"], df["predicted_label"])
        print("Précision du modèle:", accuracy)
        print(classification_report(df["label"], df["predicted_label"]))


# Interface en ligne de commande
def main():
    parser = argparse.ArgumentParser(description="Intent Classification CLI")
    parser.add_argument("--csv", type=str, help="Fichier CSV de test (mode évaluation)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de confiance pour la classification")
    args = parser.parse_args()

    classifier = IntentClassifier(threshold=args.threshold)

    if args.csv:
        classifier.evaluate(args.csv)
    else:
        print("Mode interactif. Tapez 'exit' pour quitter.")
        while True:
            text = input("Entrez une phrase: ")
            if text.lower() == "exit":
                break
            print("Prédiction:", classifier.predict_with_threshold_pkl(text))


if __name__ == "__main__":
    main()
