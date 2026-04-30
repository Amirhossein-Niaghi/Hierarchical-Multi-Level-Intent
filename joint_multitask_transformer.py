import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
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
class MultiTaskDataset(Dataset):

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
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}

        item["l1"] = torch.tensor(self.l1[idx], dtype=torch.long)
        item["l2"] = torch.tensor(self.l2[idx], dtype=torch.long)
        item["l3"] = torch.tensor(self.l3[idx], dtype=torch.long)

        return item


# ------------------------------------------------
# Joint Multi Task Model
# ------------------------------------------------
class JointMultiTaskModel(nn.Module):

    def __init__(self, model_name, n1, n2, n3, dropout=0.3):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        self.shared_dropout = nn.Dropout(dropout)

        self.head_l1 = nn.Linear(hidden, n1)
        self.head_l2 = nn.Linear(hidden, n2)
        self.head_l3 = nn.Linear(hidden, n3)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0, :]

        rep = self.shared_dropout(cls)

        out1 = self.head_l1(rep)
        out2 = self.head_l2(rep)
        out3 = self.head_l3(rep)

        return out1, out2, out3


# ------------------------------------------------
# Training
# ------------------------------------------------
def train_epoch(model, loader, optimizer, device, w1, w2, w3):

    model.train()

    ce = nn.CrossEntropyLoss()

    total_loss = 0

    for batch in loader:

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        l1 = batch["l1"].to(device)
        l2 = batch["l2"].to(device)
        l3 = batch["l3"].to(device)

        optimizer.zero_grad()

        o1, o2, o3 = model(ids, mask)

        loss1 = ce(o1, l1)
        loss2 = ce(o2, l2)
        loss3 = ce(o3, l3)

        joint_loss = w1*loss1 + w2*loss2 + w3*loss3

        joint_loss.backward()

        optimizer.step()

        total_loss += joint_loss.item()

    return total_loss / len(loader)


# ------------------------------------------------
# Evaluation
# ------------------------------------------------
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

    f1_l1 = f1_score(g1, p1, average="macro")
    f1_l2 = f1_score(g2, p2, average="macro")
    f1_l3 = f1_score(g3, p3, average="macro")

    return f1_l1, f1_l2, f1_l3


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
        epochs=12,
        w1=1.0,
        w2=1.0,
        w3=1.0
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    n1 = len(train_df["label_l1"].unique())
    n2 = len(train_df["label_l2"].unique())
    n3 = len(train_df["label_l3"].unique())

    train_ds = MultiTaskDataset(train_df, tokenizer, max_len)
    val_ds = MultiTaskDataset(val_df, tokenizer, max_len)
    test_ds = MultiTaskDataset(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = JointMultiTaskModel(model_name, n1, n2, n3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = 0
    patience = 0

    for epoch in range(epochs):

        train_epoch(model, train_loader, optimizer, device, w1, w2, w3)

        f1_l1, f1_l2, f1_l3 = evaluate(model, val_loader, device)

        score = (f1_l1 + f1_l2 + f1_l3) / 3

        if score > best_score:

            best_score = score
            torch.save(model.state_dict(), "best_joint_model.pt")
            patience = 0

        else:

            patience += 1

            if patience >= 3:
                break

    model.load_state_dict(torch.load("best_joint_model.pt"))

    f1_l1, f1_l2, f1_l3 = evaluate(model, test_loader, device)

    print("Level1 F1:", f1_l1)
    print("Level2 F1:", f1_l2)
    print("Level3 F1:", f1_l3)


if __name__ == "__main__":

    run_training()
