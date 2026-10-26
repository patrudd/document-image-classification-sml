# %% [markdown]
# # Tugas 2 SML 2026 — Document Image Classification
#
# Notebook end-to-end untuk EDA, training PyTorch, validasi stratified, interpretasi,
# dan pembuatan submission Kaggle. Target terdiri dari 16 kelas dan metrik kompetisi
# adalah accuracy. Test set tidak pernah dipakai untuk pemilihan model.

# %%
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


@dataclass(slots=True)
class Config:
    """Store all experiment settings in one reproducible object."""

    seed: int = 42
    num_classes: int = 16
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 8
    backbone_lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    val_size: float = 0.15
    patience: int = 3
    num_workers: int = 2


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
print(f"PyTorch={torch.__version__} | device={DEVICE} | work_dir={WORK_DIR}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_data_dir() -> Path:
    """Locate competition files in standard Kaggle and local paths."""

    candidates = [
        Path("/kaggle/input/document-image-classification-sd234504"),
        Path("/kaggle/input/competitions/document-image-classification-sd234504"),
        Path.cwd() / "data",
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            return candidate
    raise FileNotFoundError(
        "train.csv tidak ditemukan. Tambahkan Competition Data melalui panel Add Input."
    )


seed_everything(CFG.seed)
DATA_DIR = find_data_dir()
TRAIN_CSV = pd.read_csv(DATA_DIR / "train.csv")
TEST_CSV = pd.read_csv(DATA_DIR / "test.csv")
SAMPLE_SUBMISSION = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("DATA_DIR:", DATA_DIR)
print("train:", TRAIN_CSV.shape, "test:", TEST_CSV.shape)
display(TRAIN_CSV.head())

# %% [markdown]
# ## 1. Audit data dan distribusi kelas
#
# Pemeriksaan ini mencegah label hilang, duplikasi ID, serta ketidaksesuaian skema
# submission sebelum training yang mahal dilakukan.

# %%
assert TRAIN_CSV["id"].is_unique, "ID train harus unik."
assert TEST_CSV["id"].is_unique, "ID test harus unik."
assert set(TRAIN_CSV["label"].unique()) == set(range(CFG.num_classes))
assert list(SAMPLE_SUBMISSION.columns) == ["id", "label"]
assert len(TEST_CSV) == len(SAMPLE_SUBMISSION)

class_table = (
    TRAIN_CSV.groupby(["label", "class_name"], as_index=False)
    .size()
    .rename(columns={"size": "count"})
    .sort_values("label")
)
display(class_table)

plt.figure(figsize=(12, 5))
sns.barplot(data=class_table, x="class_name", y="count", color="#4472C4")
plt.xticks(rotation=60, ha="right")
plt.title("Distribusi kelas data training")
plt.tight_layout()
plt.show()

# %%
def resolve_image_path(data_dir: Path, filename: str) -> Path:
    """Resolve image paths whether CSV values contain folder prefixes or not."""

    raw = Path(filename)
    candidates = [data_dir / raw, data_dir / raw.name]
    if len(raw.parts) > 1:
        candidates.append(data_dir / Path(*raw.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(filename)


fig, axes = plt.subplots(4, 4, figsize=(12, 14))
for axis, (_, row) in zip(axes.flat, TRAIN_CSV.groupby("label", sort=True).first().iterrows()):
    image = Image.open(resolve_image_path(DATA_DIR, row["filename"]))
    axis.imshow(image, cmap="gray")
    axis.set_title(f"{int(row.name)} — {row['class_name']}")
    axis.axis("off")
plt.suptitle("Contoh satu dokumen per kelas", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Dataset, preprocessing, dan split
#
# `PadToSquare` mempertahankan seluruh layout halaman. Augmentasi hanya diterapkan
# pada training split. Validation menggunakan transform deterministik sehingga
# perbandingan eksperimen tetap adil dan tidak terjadi data leakage.

# %%
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Pad a document to a square canvas without cropping visual layout."""

    def __init__(self, fill: int = 255) -> None:
        """Store the white-paper padding value."""

        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        """Center a PIL image on a square canvas."""

        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        return ImageOps.expand(
            image,
            (left, top, side - width - left, side - height - top),
            fill=self.fill,
        )


class DocumentDataset(Dataset):
    """Load competition TIFF images from metadata rows."""

    def __init__(self, frame: pd.DataFrame, transform, inference: bool = False) -> None:
        """Store a defensive metadata copy and the selected transform."""

        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.inference = inference

    def __len__(self) -> int:
        """Return the number of examples."""

        return len(self.frame)

    def __getitem__(self, index: int):
        """Return a normalized tensor with its label or identifier."""

        row = self.frame.iloc[index]
        path = resolve_image_path(DATA_DIR, str(row["filename"]))
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        tensor = self.transform(image)
        target = str(row["id"]) if self.inference else int(row["label"])
        return tensor, target


common_transform = [
    PadToSquare(),
    transforms.Resize((CFG.image_size, CFG.image_size), interpolation=InterpolationMode.BICUBIC),
]
train_transform = transforms.Compose(
    common_transform
    + [
        transforms.RandomRotation(4, fill=255),
        transforms.RandomAutocontrast(p=0.25),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
eval_transform = transforms.Compose(
    common_transform + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)

train_frame, val_frame = train_test_split(
    TRAIN_CSV,
    test_size=CFG.val_size,
    random_state=CFG.seed,
    stratify=TRAIN_CSV["label"],
)
train_frame, val_frame = train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)

loader_kwargs = dict(
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    pin_memory=DEVICE.type == "cuda",
    persistent_workers=CFG.num_workers > 0,
)
train_loader = DataLoader(
    DocumentDataset(train_frame, train_transform), shuffle=True, drop_last=True, **loader_kwargs
)
val_loader = DataLoader(
    DocumentDataset(val_frame, eval_transform), shuffle=False, drop_last=False, **loader_kwargs
)
print(f"train={len(train_frame):,} | validation={len(val_frame):,}")

# %% [markdown]
# ## 3. ConvNeXt-Tiny pretrained dan training
#
# Differential learning rate menjaga backbone pretrained berubah lebih pelan daripada
# classifier baru. Bobot loss dihitung hanya dari training split, bukan validation/test.

# %%
def build_model(num_classes: int) -> nn.Module:
    """Create an ImageNet-pretrained ConvNeXt-Tiny for 16 classes."""

    try:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    except Exception as error:
        print("Bobot pretrained gagal diunduh; model dimulai dari bobot acak:", error)
        model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


model = build_model(CFG.num_classes).to(DEVICE)
head_parameters = list(model.classifier.parameters())
head_ids = {id(parameter) for parameter in head_parameters}
backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]

counts = train_frame["label"].value_counts().reindex(range(CFG.num_classes)).to_numpy()
weights = torch.tensor(len(train_frame) / (CFG.num_classes * counts), dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=CFG.label_smoothing)
optimizer = torch.optim.AdamW(
    [
        {"params": backbone_parameters, "lr": CFG.backbone_lr},
        {"params": head_parameters, "lr": CFG.head_lr},
    ],
    weight_decay=CFG.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
CHECKPOINT_PATH = WORK_DIR / "convnext_tiny_best.pt"


def run_epoch(model: nn.Module, loader: DataLoader, training: bool) -> tuple[float, float, list[int], list[int]]:
    """Run one optimization or evaluation epoch and collect predictions."""

    model.train(training)
    total_loss = 0.0
    sample_count = 0
    targets, predictions = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, desc="train" if training else "valid", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            sample_count += images.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return total_loss / sample_count, accuracy_score(targets, predictions), targets, predictions


history = []
best_accuracy, stale_epochs = -1.0, 0
for epoch in range(1, CFG.epochs + 1):
    train_loss, train_accuracy, _, _ = run_epoch(model, train_loader, training=True)
    val_loss, val_accuracy, _, _ = run_epoch(model, val_loader, training=False)
    scheduler.step()
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }
    history.append(row)
    print(
        f"Epoch {epoch:02d} | train loss/acc={train_loss:.4f}/{train_accuracy:.4f} | "
        f"val loss/acc={val_loss:.4f}/{val_accuracy:.4f}"
    )
    if val_accuracy > best_accuracy:
        best_accuracy, stale_epochs = val_accuracy, 0
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": asdict(CFG),
                "best_val_accuracy": best_accuracy,
                "history": history,
                "class_names": class_table["class_name"].tolist(),
            },
            CHECKPOINT_PATH,
        )
    else:
        stale_epochs += 1
        if stale_epochs >= CFG.patience:
            print("Early stopping.")
            break

print("Best validation accuracy:", best_accuracy)
print("Checkpoint:", CHECKPOINT_PATH)

# %% [markdown]
# ## 4. Evaluasi dan interpretasi validation split
#
# Fokuskan analisis pada kelas dengan recall rendah dan pasangan yang sering tertukar,
# bukan hanya pada satu angka accuracy.

# %%
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
val_loss, val_accuracy, y_true, y_pred = run_epoch(model, val_loader, training=False)
class_names = checkpoint["class_names"]

report = classification_report(
    y_true, y_pred, labels=list(range(CFG.num_classes)), target_names=class_names, output_dict=True, zero_division=0
)
display(pd.DataFrame(report).T.round(4))

matrix = confusion_matrix(y_true, y_pred, labels=list(range(CFG.num_classes)))
plt.figure(figsize=(13, 11))
sns.heatmap(matrix, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title(f"Confusion Matrix — validation accuracy {val_accuracy:.4f}")
plt.xticks(rotation=65, ha="right")
plt.tight_layout()
plt.savefig(WORK_DIR / "confusion_matrix.png", dpi=160)
plt.show()

history_frame = pd.DataFrame(checkpoint["history"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
history_frame.plot(x="epoch", y=["train_loss", "val_loss"], marker="o", ax=axes[0])
history_frame.plot(x="epoch", y=["train_accuracy", "val_accuracy"], marker="o", ax=axes[1])
axes[0].set_title("Loss")
axes[1].set_title("Accuracy")
plt.tight_layout()
plt.savefig(WORK_DIR / "learning_curves.png", dpi=160)
plt.show()

metrics = {
    "validation_loss": val_loss,
    "validation_accuracy": val_accuracy,
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "history": checkpoint["history"],
}
(WORK_DIR / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# %% [markdown]
# ## 5. Inferensi test dan submission
#
# Submission diurutkan kembali mengikuti `sample_submission.csv` dan divalidasi agar
# tepat 7.994 baris, tanpa ID hilang, dengan label integer 0–15.

# %%
test_loader = DataLoader(
    DocumentDataset(TEST_CSV, eval_transform, inference=True),
    batch_size=CFG.batch_size * 2,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=DEVICE.type == "cuda",
    persistent_workers=CFG.num_workers > 0,
)


@torch.inference_mode()
def predict_test(model: nn.Module, loader: DataLoader) -> pd.DataFrame:
    """Infer every test image while retaining identifiers from metadata."""

    model.eval()
    identifiers, predictions = [], []
    for images, batch_ids in tqdm(loader, desc="test"):
        logits = model(images.to(DEVICE, non_blocking=True))
        identifiers.extend(list(batch_ids))
        predictions.extend(logits.argmax(1).cpu().tolist())
    return pd.DataFrame({"id": identifiers, "label": predictions})


test_predictions = predict_test(model, test_loader)
submission = SAMPLE_SUBMISSION[["id"]].merge(
    test_predictions, on="id", how="left", validate="one_to_one"
)
assert len(submission) == len(TEST_CSV) == 7_994
assert submission["label"].notna().all()
assert submission["label"].between(0, 15).all()
submission["label"] = submission["label"].astype(int)

SUBMISSION_PATH = WORK_DIR / "submission_convnext_tiny.csv"
submission.to_csv(SUBMISSION_PATH, index=False)
display(submission.head())
print("Submission siap:", SUBMISSION_PATH)
print("Distribusi prediksi:")
display(submission["label"].value_counts().sort_index().rename("count").to_frame())

# %% [markdown]
# ## Kesimpulan eksperimen
#
# Isi setelah run selesai:
#
# - Validation accuracy terbaik: lihat output dan `validation_metrics.json`.
# - Kelas yang paling sering tertukar: baca confusion matrix, khususnya form–questionnaire,
#   scientific_publication–scientific_report, dan letter–memo.
# - Indikasi overfitting: bandingkan jarak kurva train vs validation.
# - Public score: isi hanya setelah `submission_convnext_tiny.csv` dinilai Kaggle.
# - Private score: belum tersedia sebelum kompetisi berakhir.

