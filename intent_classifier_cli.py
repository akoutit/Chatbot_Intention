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
nltk.download("stopwords")


# Classe pour la classification des intentions
class IntentClassifier:
    def __init__(self, model_path: str, label_encoder_path: str = "label_encoder.pkl"):
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.model = CamembertModel.from_pretrained("camembert-base")
        self.label_encoder = joblib.load(label_encoder_path)
        self.stop_words = set(stopwords.words("french"))
        
        if model_path.endswith(".pkl"):
            self.classifier = joblib.load(model_path)
        elif model_path.endswith(".pth"):
            self.classifier = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            raise ValueError("Format de modèle non supporté. Utilisez un fichier .pkl ou .pth")
    
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
    
    def predict(self, text: str) -> str:
        text = self.preprocess_text(text)
        embedding = self.get_embedding(text).reshape(1, -1)
        
        if isinstance(self.classifier, torch.nn.Module):
            self.classifier.eval()
            with torch.no_grad():
                output = self.classifier(torch.tensor(embedding, dtype=torch.float32))
                label_index = torch.argmax(output, dim=1).item()
        else:
            label_index = self.classifier.predict(embedding)[0]
        
        return self.label_encoder.inverse_transform([label_index])[0]
    
    def evaluate(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df["predicted_label"] = df["text"].apply(self.predict)
        accuracy = accuracy_score(df["label"], df["predicted_label"])
        print("Précision du modèle:", accuracy)
        print(classification_report(df["label"], df["predicted_label"]))

# Interface en ligne de commande
def main():
    parser = argparse.ArgumentParser(description="Intent Classification CLI")
    parser.add_argument("--model", type=str, required=True, help="Chemin du modèle de classification (.pkl ou .pth)")
    parser.add_argument("--csv", type=str, help="Fichier CSV de test (mode evaluate)")
    args = parser.parse_args()
    
    classifier = IntentClassifier(model_path=args.model)
    
    if args.csv:
        classifier.evaluate(args.csv)
    else:
        print("Mode interactif. Tapez 'exit' pour quitter.")
        while True:
            text = input("Entrez une phrase: ")
            if text.lower() == "exit":
                break
            print("Prédiction:", classifier.predict(text))

if __name__ == "__main__":
    main()