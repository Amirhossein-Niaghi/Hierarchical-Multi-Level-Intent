import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
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
# Dataset
# ------------------------------------------------------------
class HierDataset(Dataset):
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
# Top‑Down Hierarchical Classifier
# ------------------------------------------------------------
class TopDownHierClassifier(nn.Module):
    def __init__(self, model_name, sizes, embed_dim=256, dropout=0.3):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        enc_dim = self.encoder.config.hidden_size

        n1, n2, n3 = sizes

        self.dropout = nn.Dropout(dropout)

        # -------- Level 1 (root) --------
        self.h1 = nn.Linear(enc_dim, n1)

        # embed previous predictions
        self.l1_embed = nn.Embedding(n1, embed_dim)

        # -------- Level 2 --------
        self.h2 = nn.Linear(enc_dim + embed_dim, n2)

        # embed predicted level-2
        self.l2_embed = nn.Embedding(n2, embed_dim)

        # -------- Level 3 --------
        self.h3 = nn.Linear(enc_dim + embed_dim*2, n3)

    def forward(self, input_ids, attention_mask):

        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        enc = self.dropout(enc)

        # -------- L1 prediction --------
        logits1 = self.h1(enc)
        pred1 = logits1.argmax(1)
        emb1 = self.l1_embed(pred1)

        # -------- L2 prediction --------
        inp2 = torch.cat([enc, emb1], dim=1)
        logits2 = self.h2(inp2)
        pred2 = logits2.argmax(1)
        emb2 = self.l2_embed(pred2)

        # -------- L3 prediction --------
        inp3 = torch.cat([enc, emb1, emb2], dim=1)
        logits3 = self.h3(inp3)

        return logits1, logits2, logits3


# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    ce = nn.CrossEntropyLoss()
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
        L1 = ce(o1, l1)
        L2 = ce(o2, l2)
        L3 = ce(o3, l3)

        loss = L1 + L2 + L3
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate(model, loader, device):

    model.eval()
    p1, g1 = [], []
    p2, g2 = [], []
    p3, g3 = [], []

    with torch.no_grad():

        for batch in loader:

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            l1 = batch["l1"].to(device)
            l2 = batch["l2"].to(device)
            l3 = batch["l3"].to(device)

            o1, o2, o3 = model(ids, mask)

            p1.extend(o1.argmax(1).cpu().numpy())
            p2.extend(o2.argmax(1).cpu().numpy())
            p3.extend(o3.argmax(1).cpu().numpy())

            g1.extend(l1.cpu().numpy())
            g2.extend(l2.cpu().numpy())
            g3.extend(l3.cpu().numpy())

    return (
        f1_score(g1, p1, average="macro"),
        f1_score(g2, p2, average="macro"),
        f1_score(g3, p3, average="macro")
    )


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
def run_training(
        train_path="train.csv",
        val_path="val.csv",
        test_path="test.csv",
        model_name="roberta-base",
        max_len=128,
        batch_size=16,
        lr=2e-5,
        epochs=10
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    sizes = (
        train_df["label_l1"].nunique(),
        train_df["label_l2"].nunique(),
        train_df["label_l3"].nunique()
    )

    train_ds = HierDataset(train_df, tokenizer, max_len)
    val_ds = HierDataset(val_df, tokenizer, max_len)
    test_ds = HierDataset(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = TopDownHierClassifier(model_name, sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = 0
    patience = 0

    for epoch in range(epochs):

        train_epoch(model, train_loader, optimizer, device)

        f1_1, f1_2, f1_3 = evaluate(model, val_loader, device)

        score = (f1_1 + f1_2 + f1_3) / 3

        if score > best:
            best = score
            patience = 0
            torch.save(model.state_dict(), "best_topdown.pt")
        else:
            patience += 1
            if patience >= 3:
                break

    model.load_state_dict(torch.load("best_topdown.pt"))

    final = evaluate(model, test_loader, device)
    print("FINAL F1:", final)


if __name__ == "__main__":
    run_training()
