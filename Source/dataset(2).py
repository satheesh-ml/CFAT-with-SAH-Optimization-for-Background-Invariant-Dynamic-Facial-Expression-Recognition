import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# DATA
# =========================
data_dir = "archive(3)"
train_path = os.path.join(data_dir, "train")
val_path   = os.path.join(data_dir, "val")
test_path  = os.path.join(data_dir, "test")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

train_full = datasets.ImageFolder(train_path, transform=transform)
val_full   = datasets.ImageFolder(val_path, transform=transform)
test_full  = datasets.ImageFolder(test_path, transform=transform)

# limit dataset
def limit(ds, max_per_class=100):
    class_map = {i: [] for i in range(len(ds.classes))}
    for idx, (_, y) in enumerate(ds.samples):
        class_map[y].append(idx)

    selected = []
    for c, idxs in class_map.items():
        random.shuffle(idxs)
        selected += idxs[:max_per_class]

    return Subset(ds, selected)

train_ds = limit(train_full, 100)
val_ds   = limit(val_full, 100)
test_ds  = limit(test_full, 100)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

num_classes = len(train_full.classes)
print("Classes:", train_full.classes)

# =========================
# BACKBONE
# =========================
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()
backbone = backbone.to(device)

# =========================
# MODEL (UNCHANGED)
# =========================
class MGSS(nn.Module):
    def forward(self, x):
        gray = x.mean(1, keepdim=True)
        motion = torch.abs(gray - F.avg_pool2d(gray, 3, 1, 1))
        mask = torch.sigmoid(motion)
        return x * mask, x * (1 - mask)

class Causal(nn.Module):
    def forward(self, fg, bg):
        fg_v = fg.mean([2,3])
        bg_v = bg.mean([2,3])
        loss = ((bg_v - bg_v[torch.randperm(bg.size(0))])**2).mean()
        return fg_v, loss

class Head(nn.Module):
    def __init__(self, in_dim=512, num_classes=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),   # slight improvement
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class CFAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
        self.mgss = MGSS()
        self.causal = Causal()
        self.classifier = Head(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat_map = feat.unsqueeze(-1).unsqueeze(-1)

        fg, bg = self.mgss(feat_map)
        fg_vec, closs = self.causal(fg, bg)

        out = self.classifier(fg_vec)
        return out, closs

model = CFAT().to(device)

# =========================
# LOSS / OPTIMIZER
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# =========================
# TRAINING
# =========================
EPOCHS = 10

train_acc_list = []
val_acc_list = []
loss_list = []

for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        out, closs = model(x)
        loss = criterion(out, y) + 0.01 * closs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    loss_list.append(running_loss / len(train_loader))
    train_acc_list.append(train_acc)

    # ===== validation =====
    model.eval()
    v_correct, v_total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            pred = out.argmax(1)

            v_correct += (pred == y).sum().item()
            v_total += y.size(0)

    val_acc = v_correct / v_total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} | Loss {loss_list[-1]:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

# =========================
# TEST EVALUATION
# =========================
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out, _ = model(x)

        prob = torch.softmax(out, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)

        y_pred.extend(pred)
        y_true.extend(y.numpy())
        y_prob.extend(prob)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# =========================
# METRICS
# =========================
acc = accuracy_score(y_true, y_pred)

print("\nTEST ACCURACY:", acc)
print(classification_report(y_true, y_pred, target_names=train_full.classes))

cm = confusion_matrix(y_true, y_pred)

# =========================
# PLOTS (ALL SEPARATE)
# =========================

# 1. Confusion Matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=train_full.classes,
            yticklabels=train_full.classes)
plt.title("Confusion Matrix")
plt.show()

# 2. Training + Validation Accuracy
plt.figure()
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# 3. Loss Curve
plt.figure()
plt.plot(loss_list)
plt.title("Training Loss")
plt.show()

# 4. Precision-Recall Curve
plt.figure()
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

for i in range(num_classes):
    p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
    plt.plot(r, p, label=train_full.classes[i])

plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 5. ROC Curve
plt.figure()

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{train_full.classes[i]} AUC:{auc_score:.2f}")

plt.title("ROC Curve")
plt.legend()
plt.show()

# 6. FPR / FNR
fpr_list, fnr_list = [], []

for i in range(num_classes):
    tp = cm[i,i]
    fn = cm[i].sum() - tp
    fp = cm[:,i].sum() - tp
    tn = cm.sum() - (tp+fp+fn)

    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)

    fpr_list.append(fpr)
    fnr_list.append(fnr)

plt.figure()
x = np.arange(num_classes)
plt.bar(x-0.2, fpr_list, 0.4, label="FPR")
plt.bar(x+0.2, fnr_list, 0.4, label="FNR")
plt.xticks(x, train_full.classes, rotation=45)
plt.title("FPR vs FNR")
plt.legend()
plt.show()

# 7. Sample Prediction
img, label = test_ds[0]
x = img.unsqueeze(0).to(device)

with torch.no_grad():
    out, _ = model(x)
    pred = out.argmax(1).item()

img = img.permute(1,2,0).numpy()
img = (img*0.5)+0.5

plt.figure()
plt.imshow(img)
plt.title(f"True: {train_full.classes[label]} | Pred: {train_full.classes[pred]}")
plt.axis("off")
plt.show()