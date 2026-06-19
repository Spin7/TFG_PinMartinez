import os
import torch
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
DATASET_ROOT = "dataset_cls"
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 50
PATIENCE = 10
IMAGE_SIZE = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "best_model_fixed.pth"

# =========================
# NORMALIZATION
# =========================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# =========================
# TRANSFORMS (CLEAN + CONTROLLED)
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# =========================
# DATA
# =========================
train_data = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=train_transform)
val_data   = datasets.ImageFolder(os.path.join(DATASET_ROOT, "valid"), transform=val_transform)
test_data  = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"), transform=val_transform)

class_names = train_data.classes
print("Classes:", class_names)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# CLASS WEIGHTS (better than sampler)
# =========================
targets = [label for _, label in train_data.samples]
class_counts = np.bincount(targets)
weights = 1.0 / class_counts
weights = weights / weights.sum()

class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# =========================
# MODEL (ACCURACY FOCUSED)
# =========================
model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, len(class_names))

model = model.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# TRAIN
# =========================
def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# =========================
# TRAIN LOOP
# =========================
best_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()
    val_loss, val_acc = evaluate(val_loader)

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✔ Best model saved (by accuracy)")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping")
        break

# =========================
# TEST
# =========================
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n===== TEST RESULTS =====")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))