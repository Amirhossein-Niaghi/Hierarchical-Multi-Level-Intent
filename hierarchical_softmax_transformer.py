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
# Hierarchical Softmax Model
# ------------------------------------------------------------
class HierSoftmaxTransformer(nn.Module):
    def __init__(self, model_name, sizes, hmap_l2, hmap_l3, dropout=0.3):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        enc_dim = self.encoder.config.hidden_size

        self.n1, self.n2, self.n3 = sizes
        self.dropout = nn.Dropout(dropout)

        self.h1 = nn.Linear(enc_dim, self.n1)
        self.h2 = nn.Linear(enc_dim, self.n2)
        self.h3 = nn.Linear(enc_dim, self.n3)

        # hierarchy maps
        self.hmap_l2 = hmap_l2  # dict: l1 -> [valid l2]
        self.hmap_l3 = hmap_l3  # dict: l2 -> [valid l3]

    def masked_softmax(self, logits, mask):
        logits = logits.masked_fill(mask == 0, float("-inf"))
        return torch.log_softmax(logits, dim=1)

    def forward(self, input_ids, attention_mask, y1=None, y2=None):

        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        enc = self.dropout(enc)

        # -------- Level 1 --------
        logits1 = self.h1(enc)
        logp1 = torch.log_softmax(logits1, dim=1)
        pred1 = logits1.argmax(1) if y1 is None else y1

        # -------- Level 2 --------
        logits2 = self.h2(enc)
        mask2 = torch.zeros_like(logits2)

        for i, p in enumerate(pred1.tolist()):
            mask2[i, self.hmap_l2[p]] = 1

        logp2 = self.masked_softmax(logits2, mask2)
        pred2 = logits2.argmax(1) if y2 is None else y2

        # -------- Level 3 --------
        logits3 = self.h3(enc)
        mask3 = torch.zeros_like(logits3)

        for i, p in enumerate(pred2.tolist()):
            mask3[i, self.hmap_l3[p]] = 1

        logp3 = self.masked_softmax(logits3, mask3)

        return logp1, logp2, logp3


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0

    for batch in loader:

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        y1 = batch["l1"].to(device)
        y2 = batch["l2"].to(device)
        y3 = batch["l3"].to(device)

        optimizer.zero_grad()

        logp1, logp2, logp3 = model(ids, mask, y1, y2)

        loss = (
            nn.NLLLoss()(logp1, y1) +
            nn.NLLLoss()(logp2, y2) +
            nn.NLLLoss()(logp3, y3)
        )

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

            y1 = batch["l1"].to(device)
            y2 = batch["l2"].to(device)
            y3 = batch["l3"].to(device)

            logp1, logp2, logp3 = model(ids, mask)

            p1.extend(logp1.argmax(1).cpu().numpy())
            p2.extend(logp2.argmax(1).cpu().numpy())
            p3.extend(logp3.argmax(1).cpu().numpy())

            g1.extend(y1.cpu().numpy())
            g2.extend(y2.cpu().numpy())
            g3.extend(y3.cpu().numpy())

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
        train_df.label_l1.nunique(),
        train_df.label_l2.nunique(),
        train_df.label_l3.nunique()
    )

    # build hierarchy maps
    hmap_l2 = train_df.groupby("label_l1")["label_l2"].unique().to_dict()
    hmap_l3 = train_df.groupby("label_l2")["label_l3"].unique().to_dict()

    train_ds = HierDataset(train_df, tokenizer, max_len)
    val_ds = HierDataset(val_df, tokenizer, max_len)
    test_ds = HierDataset(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = HierSoftmaxTransformer(
        model_name, sizes, hmap_l2, hmap_l3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best, patience = 0, 0

    for _ in range(epochs):

        train_epoch(model, train_loader, optimizer, device)
        f1s = evaluate(model, val_loader, device)
        score = sum(f1s) / 3

        if score > best:
            best = score
            patience = 0
            torch.save(model.state_dict(), "best_hsoftmax.pt")
        else:
            patience += 1
            if patience >= 3:
                break

    model.load_state_dict(torch.load("best_hsoftmax.pt"))
    print("FINAL F1:", evaluate(model, test_loader, device))


if __name__ == "__main__":
    run_training()
