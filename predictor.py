# predictor.py

import json
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


class MentalHealthRiskPredictor:
    def __init__(self, artifact_dir="deployment_artifacts"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = f"{artifact_dir}/best_model"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        ).to(self.device)
        self.classifier_model.eval()

        self.encoder_model = AutoModel.from_pretrained(
            self.model_path
        ).to(self.device)
        self.encoder_model.eval()

        self.kmeans = joblib.load(f"{artifact_dir}/final_kmeans.pkl")
        self.reducer = joblib.load(f"{artifact_dir}/umap_reducer.pkl")
        self.cluster_name_map = joblib.load(f"{artifact_dir}/cluster_name_map.pkl")

        with open(f"{artifact_dir}/risk_config.json", "r") as f:
            self.risk_config = json.load(f)

        self.emotion_columns = self.risk_config["emotion_columns"]
        self.risk_weights = self.risk_config["risk_weights"]

    def assign_risk_level(self, score):
        if score < 2.0:
            return "Mild Risk"
        elif score < 4.0:
            return "Low-Moderate Risk"
        elif score < 6.0:
            return "Moderate Risk"
        elif score < 8.0:
            return "High Risk"
        else:
            return "Very High Risk"

    def predict(self, text, threshold=0.5):
        inputs = self.tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        emotion_probabilities = {
            self.emotion_columns[i]: float(probs[i])
            for i in range(len(self.emotion_columns))
        }

        predicted_emotions = [
            self.emotion_columns[i]
            for i in range(len(self.emotion_columns))
            if probs[i] >= threshold
        ]

        risk_score = sum(
            emotion_probabilities[e] * self.risk_weights[e]
            for e in self.emotion_columns
        )

        with torch.no_grad():
            encoder_outputs = self.encoder_model(**inputs)

        cls_embedding = encoder_outputs.last_hidden_state[:, 0, :].cpu().numpy()
        reduced_embedding = self.reducer.transform(cls_embedding)

        cluster_id = int(self.kmeans.predict(reduced_embedding)[0])
        cluster_name = self.cluster_name_map.get(str(cluster_id), None)

        if cluster_name is None:
            cluster_name = self.cluster_name_map.get(cluster_id, "Unknown Cluster")

        return {
            "input_text": text,
            "predicted_emotions": predicted_emotions,
            "emotion_probabilities": emotion_probabilities,
            "risk_score": round(float(risk_score), 4),
            "risk_level": self.assign_risk_level(risk_score),
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "confidence": round(float(np.max(probs)), 4)
        }