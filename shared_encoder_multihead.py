import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# Dataset Handling: dataset must contain 3 label columns
# ------------------------------------------------------------
class MultiLevelDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.l1 = df["label_l1"].tolist()
        self.l2 = df["label_l2"].tolist()
        self.l3 = df["label_l3"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}

        item["l1"] = torch.tensor(self.l1[idx], dtype=torch.long)
        item["l2"] = torch.tensor(self.l2[idx], dtype=torch.long)
        item["l3"] = torch.tensor(self.l3[idx], dtype=torch.long)

        return item

# ------------------------------------------------------------
# Shared Encoder + Three Independent Heads
# ------------------------------------------------------------
class SharedEncoderMultiHead(nn.Module):
    def __init__(self, model_name, num_l1, num_l2, num_l3, dropout=0.3):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.head1 = nn.Linear(hidden, num_l1)
        self.head2 = nn.Linear(hidden, num_l2)
        self.head3 = nn.Linear(hidden, num_l3)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        logits1 = self.head1(cls)
        logits2 = self.head2(cls)
        logits3 = self.head3(cls)

        return logits1, logits2, logits3

# ------------------------------------------------------------
# Training Epoch
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0

    for batch in loader:

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        l1 = batch["l1"].to(device)
        l2 = batch["l2"].to(device)
        l3 = batch["l3"].to(device)

        optimizer.zero_grad()

        o1, o2, o3 = model(ids, mask)

        loss1 = criterion(o1, l1)
        loss2 = criterion(o2, l2)
        loss3 = criterion(o3, l3)

        loss = loss1 + loss2 + loss3  # equal weights

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate(model, loader, device):

    model.eval()

    preds1, gold1 = [], []
    preds2, gold2 = [], []
    preds3, gold3 = [], []

    with torch.no_grad():

        for batch in loader:

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            l1 = batch["l1"].to(device)
            l2 = batch["l2"].to(device)
            l3 = batch["l3"].to(device)

            o1, o2, o3 = model(ids, mask)

            p1 = o1.argmax(dim=1).cpu().numpy()
            p2 = o2.argmax(dim=1).cpu().numpy()
            p3 = o3.argmax(dim=1).cpu().numpy()

            preds1.extend(p1); gold1.extend(l1.cpu().numpy())
            preds2.extend(p2); gold2.extend(l2.cpu().numpy())
            preds3.extend(p3); gold3.extend(l3.cpu().numpy())

    f1_1 = f1_score(gold1, preds1, average="macro")
    f1_2 = f1_score(gold2, preds2, average="macro")
    f1_3 = f1_score(gold3, preds3, average="macro")

    return f1_1, f1_2, f1_3

# ------------------------------------------------------------
# Training Pipeline
# ------------------------------------------------------------
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

    num_l1 = len(train_df["label_l1"].unique())
    num_l2 = len(train_df["label_l2"].unique())
    num_l3 = len(train_df["label_l3"].unique())

    train_ds = MultiLevelDataset(train_df, tokenizer, max_len)
    val_ds = MultiLevelDataset(val_df, tokenizer, max_len)
    test_ds = MultiLevelDataset(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SharedEncoderMultiHead(model_name, num_l1, num_l2, num_l3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_score = 0
    patience = 0
    patience_limit = 3

    for epoch in range(epochs):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        f1_l1, f1_l2, f1_l3 = evaluate(model, val_loader, device)

        val_score = (f1_l1 + f1_l2 + f1_l3) / 3

        if val_score > best_score:
            torch.save(model.state_dict(), "best_shared_encoder.pt")
            best_score = val_score
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break

    model.load_state_dict(torch.load("best_shared_encoder.pt"))

    test_f1_l1, test_f1_l2, test_f1_l3 = evaluate(model, test_loader, device)

    print("\nTEST RESULTS:")
    print("Level‑1 Macro F1:", test_f1_l1)
    print("Level‑2 Macro F1:", test_f1_l2)
    print("Level‑3 Macro F1:", test_f1_l3)

# ------------------------------------------------------------
if __name__ == "__main__":
    run_training()
