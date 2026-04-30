import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class DAMTDataset(Dataset):

    def __init__(self, df, tokenizer, max_len=128):

        self.texts = df["text"].tolist()
        self.l1 = df["label_l1"].tolist()
        self.l2 = df["label_l2"].tolist()
        self.l3 = df["label_l3"].tolist()

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def augment(self, text):
        words = text.split()
        if len(words) > 3:
            i = random.randint(0, len(words)-1)
            words.pop(i)
        return " ".join(words)

    def __getitem__(self, idx):

        text = self.texts[idx]
        text_aug = self.augment(text)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        enc_aug = self.tokenizer(
            text_aug,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["input_ids_aug"] = enc_aug["input_ids"].squeeze(0)
        item["attention_mask_aug"] = enc_aug["attention_mask"].squeeze(0)

        item["l1"] = torch.tensor(self.l1[idx])
        item["l2"] = torch.tensor(self.l2[idx])
        item["l3"] = torch.tensor(self.l3[idx])

        return item


# ---------------------------------------------------------
# DAMT MODEL
# ---------------------------------------------------------
class DAMT(nn.Module):

    def __init__(self, model_name, sizes, dropout=0.3):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        n1, n2, n3 = sizes

        self.dropout = nn.Dropout(dropout)

        # Level 1
        self.head1 = nn.Linear(hidden, n1)

        # Dependency projection
        self.dep12 = nn.Linear(n1, hidden)
        self.dep23 = nn.Linear(n2, hidden)

        # Level 2
        self.head2 = nn.Linear(hidden*2, n2)

        # Level 3
        self.head3 = nn.Linear(hidden*3, n3)

    def encode(self, input_ids, mask):

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=mask
        )

        cls = out.last_hidden_state[:, 0]
        return self.dropout(cls)

    def forward(self, ids, mask):

        h_cls = self.encode(ids, mask)

        # -------- Level 1 --------
        logits1 = self.head1(h_cls)
        p1 = F.softmax(logits1, dim=1)

        # dependency representation z2
        dep1 = self.dep12(p1)

        z2 = torch.cat([h_cls, dep1], dim=1)

        # -------- Level 2 --------
        logits2 = self.head2(z2)
        p2 = F.softmax(logits2, dim=1)

        dep2 = self.dep23(p2)

        z3 = torch.cat([h_cls, dep1, dep2], dim=1)

        # -------- Level 3 --------
        logits3 = self.head3(z3)

        return logits1, logits2, logits3, h_cls


# ---------------------------------------------------------
# Contrastive Loss
# ---------------------------------------------------------
def contrastive_loss(z, z_aug, tau=0.07):

    z = F.normalize(z, dim=1)
    z_aug = F.normalize(z_aug, dim=1)

    sim = torch.mm(z, z_aug.t()) / tau

    labels = torch.arange(z.size(0)).to(z.device)

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_epoch(model, loader, optimizer, device,
                lambdas=(1,1,1), gamma=0.5, alpha=0.5):

    model.train()

    ce = nn.CrossEntropyLoss()

    total = 0

    for batch in loader:

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        ids_aug = batch["input_ids_aug"].to(device)
        mask_aug = batch["attention_mask_aug"].to(device)

        y1 = batch["l1"].to(device)
        y2 = batch["l2"].to(device)
        y3 = batch["l3"].to(device)

        optimizer.zero_grad()

        o1,o2,o3,z = model(ids,mask)
        _,_,_,z_aug = model(ids_aug,mask_aug)

        L1 = ce(o1,y1)
        L2 = ce(o2,y2)
        L3 = ce(o3,y3)

        L_contrast = contrastive_loss(z,z_aug)

        loss = (
            lambdas[0]*L1 +
            lambdas[1]*L2 +
            lambdas[2]*L3 +
            gamma*L_contrast +
            alpha*L_contrast
        )

        loss.backward()
        optimizer.step()

        total += loss.item()

    return total/len(loader)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate(model, loader, device):

    model.eval()

    p1,g1=[],[]
    p2,g2=[],[]
    p3,g3=[],[]

    with torch.no_grad():

        for batch in loader:

            ids=batch["input_ids"].to(device)
            mask=batch["attention_mask"].to(device)

            y1=batch["l1"].to(device)
            y2=batch["l2"].to(device)
            y3=batch["l3"].to(device)

            o1,o2,o3,_ = model(ids,mask)

            p1+=o1.argmax(1).cpu().tolist()
            p2+=o2.argmax(1).cpu().tolist()
            p3+=o3.argmax(1).cpu().tolist()

            g1+=y1.cpu().tolist()
            g2+=y2.cpu().tolist()
            g3+=y3.cpu().tolist()

    return (
        f1_score(g1,p1,average="macro"),
        f1_score(g2,p2,average="macro"),
        f1_score(g3,p3,average="macro")
    )


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------
def run_training(
        train_path="train.csv",
        val_path="val.csv",
        test_path="test.csv",
        model_name="roberta-base",
        batch_size=16,
        max_len=128,
        lr=2e-5,
        epochs=10):

    device="cuda" if torch.cuda.is_available() else "cpu"

    tokenizer=AutoTokenizer.from_pretrained(model_name)

    train=pd.read_csv(train_path)
    val=pd.read_csv(val_path)
    test=pd.read_csv(test_path)

    sizes=(
        train.label_l1.nunique(),
        train.label_l2.nunique(),
        train.label_l3.nunique()
    )

    train_ds=DAMTDataset(train,tokenizer,max_len)
    val_ds=DAMTDataset(val,tokenizer,max_len)
    test_ds=DAMTDataset(test,tokenizer,max_len)

    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=batch_size)
    test_loader=DataLoader(test_ds,batch_size=batch_size)

    model=DAMT(model_name,sizes).to(device)

    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    best=0
    patience=0

    for epoch in range(epochs):

        train_epoch(model,train_loader,optimizer,device)

        f1s=evaluate(model,val_loader,device)
        score=sum(f1s)/3

        if score>best:
            best=score
            patience=0
            torch.save(model.state_dict(),"best_damt.pt")

        else:
            patience+=1
            if patience>=3:
                break

    model.load_state_dict(torch.load("best_damt.pt"))

    print("FINAL TEST F1:",evaluate(model,test_loader,device))


if __name__=="__main__":
    run_training()
