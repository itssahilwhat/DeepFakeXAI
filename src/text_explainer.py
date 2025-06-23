import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

class ArtifactTextExplainer:
    def __init__(self, model_path=None, train_dataset=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        if model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            if train_dataset:
                training_args = TrainingArguments(
                    output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16,
                    save_steps=500, save_total_limit=2
                )
                trainer = Trainer(model=self.model, args=training_args, train_dataset=train_dataset)
                trainer.train()
        self.model.to(self.device)
        self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.eval()

    def predict(self, artifacts):
        inputs = self.tokenizer(artifacts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        return logits.cpu().numpy(), probs.cpu().numpy()

    def explain(self, artifact_list):
        explanations = []
        for art in artifact_list:
            if "blurr" in art:
                explanations.append(f"The image contains a blur artifact in region: {art}.")
            elif "warp" in art or "distort" in art:
                explanations.append(f"The image contains a warp/distortion artifact: {art}.")
            else:
                explanations.append(f"Detected artifact: {art}.")
        return explanations