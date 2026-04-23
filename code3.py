!pip uninstall -y timm -q
!pip install timm==0.6.13 einops scikit-learn -q

import os, json, zipfile
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import timm
from sklearn.model_selection import KFold

DATASET_PATH = "/content/ALD-E-ImageMiner/icdar2026-competition-data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 384
BATCH_SIZE = 6
EPOCHS = 8
def extract_samples(dataset_path, split="train"):
    samples = []
    split_path = os.path.join(dataset_path, split)

    for domain in os.listdir(split_path):
        dpath = os.path.join(split_path, domain)
        if not os.path.isdir(dpath): continue

        for usecase in os.listdir(dpath):
            upath = os.path.join(dpath, usecase)
            if not os.path.isdir(upath): continue

            for paper in os.listdir(upath):
                ppath = os.path.join(upath, paper)
                if not os.path.isdir(ppath): continue

                base = os.path.join(ppath, "images")
                if not os.path.exists(base): continue

                for f in os.listdir(base):
                    if not f.endswith(".json"): continue

                    data = json.load(open(os.path.join(base,f)))

                    img_name = f.replace(".json","")
                    img_path = None
                    for ext in [".jpg",".png",".jpeg"]:
                        p = os.path.join(base, img_name+ext)
                        if os.path.exists(p):
                            img_path = p
                            break

                    if img_path is None: continue

                    for k,b in data.get("bbox",{}).items():
                        entry = {
                            "img_path": img_path,
                            "bbox": b,
                            "subfig": k,
                            "sample_id": data["sample_id"]
                        }

                        if split != "test":
                            label = data.get("classification",{}).get(k,"").lower()
                            if label == "": continue
                            entry["label"] = label

                        samples.append(entry)

    return samples
class SquarePad:
    def __call__(self, image):
        w,h = image.size
        m = max(w,h)
        return TF.pad(image, ((m-w)//2,(m-h)//2,(m-w)//2,(m-h)//2), 255)

train_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),  # light only

    transforms.ColorJitter(0.1,0.1,0.1,0.05),

    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

test_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])
class FigureDataset(Dataset):
    def __init__(self, samples, transform=None, label2idx=None):
        self.samples = samples
        self.transform = transform

        if label2idx is None:
            labels = sorted(set(s["label"] for s in samples if "label" in s))
            self.label2idx = {l:i for i,l in enumerate(labels)}
        else:
            self.label2idx = label2idx

        self.idx2label = {i:l for l,i in self.label2idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        b = s["bbox"]

        crop = img.crop((b["x"], b["y"], b["x"]+b["width"], b["y"]+b["height"]))

        if self.transform:
            crop = self.transform(crop)

        label = self.label2idx.get(s.get("label",""),0)
        return crop, label
train_samples = extract_samples(DATASET_PATH,"train")

labels = sorted(set(s["label"] for s in train_samples))
label2idx = {l:i for i,l in enumerate(labels)}
NUM_CLASSES = len(label2idx)
def train_fold(train_data, val_data, fold):

    train_ds = FigureDataset(train_data, train_transform, label2idx)
    val_ds = FigureDataset(val_data, test_transform, label2idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = timm.create_model('convnext_base_384_in22ft1k',
                              pretrained=True,
                              num_classes=NUM_CLASSES).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    best = 0

    for epoch in range(EPOCHS):
        model.train()

        for x,y in train_loader:
            x,y = x.to(device),y.to(device)

            optimizer.zero_grad()
            out = model(x)

            loss = F.cross_entropy(out,y, label_smoothing=0.05)

            loss.backward()
            optimizer.step()

        model.eval()
        correct,total = 0,0

        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device),y.to(device)
                pred = model(x).argmax(1)

                correct += (pred==y).sum().item()
                total += y.size(0)

        acc = correct/total
        print(f"Fold {fold} Epoch {epoch+1}: {acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), f"fold{fold}.pth")
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_samples)):

    train_data = [train_samples[i] for i in train_idx]
    val_data = [train_samples[i] for i in val_idx]

    train_fold(train_data, val_data, fold)
def tta(model, x):
    outs = []

    for flip in [False, True]:
        img = x.clone()
        if flip:
            img = torch.flip(img, dims=[3])

        outs.append(F.softmax(model(img), dim=1))

    return torch.stack(outs).mean(0)

models = []

for i in range(3):
    m = timm.create_model('convnext_base_384_in22ft1k',
                          num_classes=NUM_CLASSES).to(device)
    m.load_state_dict(torch.load(f"fold{i}.pth"))
    m.eval()
    models.append(m)
def generate_submission():
    test_samples = extract_samples(DATASET_PATH,"test")

    res = {}

    for s in test_samples:
        img = Image.open(s["img_path"]).convert("RGB")
        b = s["bbox"]

        crop = img.crop((b["x"],b["y"],b["x"]+b["width"],b["y"]+b["height"]))
        x = test_transform(crop).unsqueeze(0).to(device)

        prob = 0

        for m in models:
            prob += tta(m, x)

        prob /= len(models)

        pred = prob.argmax(1).item()

        sid = s["sample_id"]

        if sid not in res:
            res[sid] = {"sample_id": sid, "classification": {}}

        res[sid]["classification"][s["subfig"]] = list(label2idx.keys())[pred]

    with open("prediction_data.json","w") as f:
        json.dump(list(res.values()), f, indent=4)

    with zipfile.ZipFile("submission.zip","w") as z:
        z.write("prediction_data.json")

    print("✅ DONE")

generate_submission()