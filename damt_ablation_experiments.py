import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def __init__(self, df, tokenizer, max_len=128, augment=False):

        self.texts = df["text"].tolist()
        self.l1 = df["label_l1"].tolist()
        self.l2 = df["label_l2"].tolist()
        self.l3 = df["label_l3"].tolist()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment_flag = augment

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

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k:v.squeeze(0) for k,v in enc.items()}

        item["l1"] = torch.tensor(self.l1[idx])
        item["l2"] = torch.tensor(self.l2[idx])
        item["l3"] = torch.tensor(self.l3[idx])

        if self.augment_flag:

            aug = self.augment(text)

            enc2 = self.tokenizer(
                aug,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )

            item["ids_aug"] = enc2["input_ids"].squeeze(0)
            item["mask_aug"] = enc2["attention_mask"].squeeze(0)

        return item


# ------------------------------------------------
# Flat Transformer (Single Task)
# ------------------------------------------------

class FlatTransformer(nn.Module):

    def __init__(self, model_name, num_labels):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, ids, mask):

        out = self.encoder(ids, attention_mask=mask)

        cls = out.last_hidden_state[:,0]

        return self.fc(cls)


# ------------------------------------------------
# Parallel Multi Task Transformer
# ------------------------------------------------

class ParallelMTL(nn.Module):

    def __init__(self, model_name, sizes):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        n1,n2,n3 = sizes

        self.h1 = nn.Linear(hidden,n1)
        self.h2 = nn.Linear(hidden,n2)
        self.h3 = nn.Linear(hidden,n3)

    def forward(self, ids, mask):

        out = self.encoder(ids, attention_mask=mask)

        cls = out.last_hidden_state[:,0]

        return self.h1(cls), self.h2(cls), self.h3(cls)


# ------------------------------------------------
# Hierarchical without Dependency
# ------------------------------------------------

class HierarchicalNoDependency(nn.Module):

    def __init__(self, model_name, sizes):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        n1,n2,n3 = sizes

        self.h1 = nn.Linear(hidden,n1)
        self.h2 = nn.Linear(hidden,n2)
        self.h3 = nn.Linear(hidden,n3)

    def forward(self, ids, mask):

        out = self.encoder(ids, attention_mask=mask)

        cls = out.last_hidden_state[:,0]

        l1 = self.h1(cls)
        l2 = self.h2(cls)
        l3 = self.h3(cls)

        return l1,l2,l3


# ------------------------------------------------
# DAMT without Contrastive
# ------------------------------------------------

class DAMT_NoContrast(nn.Module):

    def __init__(self, model_name, sizes):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        n1,n2,n3 = sizes

        self.h1 = nn.Linear(hidden,n1)

        self.dep12 = nn.Linear(n1,hidden)
        self.dep23 = nn.Linear(n2,hidden)

        self.h2 = nn.Linear(hidden*2,n2)
        self.h3 = nn.Linear(hidden*3,n3)

    def forward(self, ids, mask):

        out = self.encoder(ids, attention_mask=mask)

        cls = out.last_hidden_state[:,0]

        o1 = self.h1(cls)

        p1 = F.softmax(o1,dim=1)
        d1 = self.dep12(p1)

        z2 = torch.cat([cls,d1],1)

        o2 = self.h2(z2)

        p2 = F.softmax(o2,dim=1)
        d2 = self.dep23(p2)

        z3 = torch.cat([cls,d1,d2],1)

        o3 = self.h3(z3)

        return o1,o2,o3


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

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

            o1,o2,o3=model(ids,mask)

            p1+=o1.argmax(1).cpu().tolist()
            p2+=o2.argmax(1).cpu().tolist()
            p3+=o3.argmax(1).cpu().tolist()

            g1+=y1.cpu().tolist()
            g2+=y2.cpu().tolist()
            g3+=y3.cpu().tolist()

    f1_1=f1_score(g1,p1,average="macro")
    f1_2=f1_score(g2,p2,average="macro")
    f1_3=f1_score(g3,p3,average="macro")

    return f1_1,f1_2,f1_3,(f1_1+f1_2+f1_3)/3


# ------------------------------------------------
# Generic Training
# ------------------------------------------------

def train(model, loader, optimizer, device):

    ce=nn.CrossEntropyLoss()

    model.train()

    for batch in loader:

        ids=batch["input_ids"].to(device)
        mask=batch["attention_mask"].to(device)

        y1=batch["l1"].to(device)
        y2=batch["l2"].to(device)
        y3=batch["l3"].to(device)

        optimizer.zero_grad()

        o1,o2,o3=model(ids,mask)

        loss=ce(o1,y1)+ce(o2,y2)+ce(o3,y3)

        loss.backward()

        optimizer.step()


# ------------------------------------------------
# Run Ablation Experiment
# ------------------------------------------------

def run_ablation(model_type):

    device="cuda" if torch.cuda.is_available() else "cpu"

    tokenizer=AutoTokenizer.from_pretrained("roberta-base")

    train=pd.read_csv("train.csv")
    val=pd.read_csv("val.csv")
    test=pd.read_csv("test.csv")

    sizes=(
        train.label_l1.nunique(),
        train.label_l2.nunique(),
        train.label_l3.nunique()
    )

    train_ds=IntentDataset(train,tokenizer)
    val_ds=IntentDataset(val,tokenizer)
    test_ds=IntentDataset(test,tokenizer)

    train_loader=DataLoader(train_ds,batch_size=16,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=16)
    test_loader=DataLoader(test_ds,batch_size=16)

    if model_type=="flat":

        model=FlatTransformer("roberta-base",sizes[2])

    elif model_type=="parallel":

        model=ParallelMTL("roberta-base",sizes)

    elif model_type=="hier_no_dep":

        model=HierarchicalNoDependency("roberta-base",sizes)

    elif model_type=="damt_no_contrast":

        model=DAMT_NoContrast("roberta-base",sizes)

    model=model.to(device)

    optimizer=torch.optim.Adam(model.parameters(),lr=2e-5)

    for epoch in range(10):

        train(model,train_loader,optimizer,device)

    print(model_type,evaluate(model,test_loader,device))


if __name__=="__main__":

    run_ablation("flat")
    run_ablation("parallel")
    run_ablation("hier_no_dep")
    run_ablation("damt_no_contrast")
