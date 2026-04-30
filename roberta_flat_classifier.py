import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------
# Reproducibility
# ------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------
# Dataset
# ------------------------------------------------
class IntentDataset(Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ------------------------------------------------
# Model
# ------------------------------------------------
class RobertaIntentClassifier(nn.Module):

    def __init__(self, model_name, num_labels, dropout=0.3):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0, :]

        cls = self.dropout(cls)

        logits = self.classifier(cls)

        return logits


# ------------------------------------------------
# Training
# ------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, mask)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------------------------
# Evaluation
# ------------------------------------------------
def evaluate(model, loader, device):

    model.eval()

    preds = []
    gold = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, mask)

            p = logits.argmax(dim=1)

            preds.extend(p.cpu().numpy())
            gold.extend(labels.cpu().numpy())

    macro_f1 = f1_score(gold, preds, average="macro")
    acc = accuracy_score(gold, preds)

    report = classification_report(gold, preds, digits=4)

    return macro_f1, acc, report


# ------------------------------------------------
# Pipeline
# ------------------------------------------------
def run_training(

        train_path="train.csv",
        val_path="val.csv",
        test_path="test.csv",
        model_name="roberta-base",
        max_len=128,
        batch_size=16,
        lr=2e-5,
        epochs=12
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    num_labels = len(train_df["label"].unique())

    train_ds = IntentDataset(train_df, tokenizer, max_len)
    val_ds = IntentDataset(val_df, tokenizer, max_len)
    test_ds = IntentDataset(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = RobertaIntentClassifier(model_name, num_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience = 0
    patience_limit = 3

    for epoch in range(epochs):

        loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_f1, _, _ = evaluate(model, val_loader, device)

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(model.state_dict(), "best_roberta.pt")

            patience = 0

        else:

            patience += 1

            if patience >= patience_limit:
                break

    model.load_state_dict(torch.load("best_roberta.pt"))

    macro_f1, acc, report = evaluate(model, test_loader, device)

    print("\nFINAL TEST RESULTS")
    print("Macro F1:", macro_f1)
    print("Accuracy:", acc)
    print(report)


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    run_training()
