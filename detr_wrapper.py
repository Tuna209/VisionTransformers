"""
DETR Fine-tuning on AU-AIR Dataset - Simplified Version

This script fine-tunes a DETR (DEtection TRansformer) model on the AU-AIR dataset.
It handles data loading, model training, evaluation, and visualization in a single file.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import wandb
from datetime import datetime
from torchvision import transforms as T
from torchvision.transforms import functional as F
from transformers import DetrForObjectDetection, DetrConfig

# Configuration
DATASET_ROOT = "dataset"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
ANNOTATIONS_FN = os.path.join(DATASET_ROOT, "annotations.json")

# Model / training
NUM_CLASSES = 8
CLASS_NAMES = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
IMAGE_SIZE = (800, 800)
BATCH_SIZE = 4
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
CLIP_MAX_NORM = 0.1

# Optimizer / scheduler
BETA1, BETA2 = 0.9, 0.999
LR_SCHED_TMAX = NUM_EPOCHS
LR_SCHED_ETA_MIN = 1e-6

# Checkpoint & logs
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQ = 1
WANDB_PROJECT = "DETR"
WANDB_ENTITY = "tuna-ozturk1283"
EXP_NAME = f"detr-auair-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Eval & viz
CONF_THRESH = 0.5
IOU_THRESH = 0.5
VIZ_OUTPUT_DIR = "visualizations"
NUM_VIZ_IMAGES = 10

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Initialize wandb
def init_wandb(config_dict=None):
    """Initialize Weights & Biases logging."""
    if config_dict is None:
        config_dict = {
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": f"AdamW (beta1={BETA1}, beta2={BETA2})",
            "scheduler": f"CosineAnnealingLR (T_max={LR_SCHED_TMAX}, eta_min={LR_SCHED_ETA_MIN})",
            "clip_max_norm": CLIP_MAX_NORM,
            "confidence_threshold": CONF_THRESH,
            "iou_threshold": IOU_THRESH,
        }

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=EXP_NAME,
        config=config_dict
    )

# Load annotations
def load_annotations(fn=ANNOTATIONS_FN):
    """Read the JSON annotations file."""
    with open(fn) as f:
        return json.load(f)

# Create data splits
def create_splits(ann, train=0.7, val=0.15, test=0.15, seed=42):
    """Shuffle & split annotations into train/val/test."""
    assert abs(train+val+test-1.0) < 1e-5
    random.seed(seed)
    all_ann = ann["annotations"]
    random.shuffle(all_ann)
    n = len(all_ann)
    t = int(n*train)
    v = t + int(n*val)
    print(f"Splits: {t} train, {v-t} val, {n-v} test")
    return all_ann[:t], all_ann[t:v], all_ann[v:]

# AU-AIR Dataset
class AUAIRDataset(Dataset):
    """PyTorch Dataset for AU-AIR."""
    def __init__(self, anns, img_dir, transform=None):
        self.anns = anns
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): 
        return len(self.anns)

    def __getitem__(self, i):
        a = self.anns[i]
        img_path = os.path.join(self.img_dir, a["image_name"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IOError):
            # Skip unavailable images by returning a random valid image
            random_idx = random.randint(0, len(self.anns) - 1)
            while random_idx == i:
                random_idx = random.randint(0, len(self.anns) - 1)
            return self.__getitem__(random_idx)
            
        w = a.get("image_width", 1920)
        h = a.get("image_height", 1080)

        boxes, labels = [], []
        for b in a["bbox"]:
            x0 = b["left"]/w
            y0 = b["top"]/h
            x1 = (b["left"]+b["width"])/w
            y1 = (b["top"]+b["height"])/h
            
            # Skip invalid boxes
            if x1 <= x0 or y1 <= y0:
                continue
                
            boxes.append([x0, y0, x1, y1])
            labels.append(b["class"])
            
        # If no valid boxes, return a random image
        if len(boxes) == 0:
            random_idx = random.randint(0, len(self.anns) - 1)
            while random_idx == i:
                random_idx = random.randint(0, len(self.anns) - 1)
            return self.__getitem__(random_idx)
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([i]),
            "orig_size": torch.tensor([h, w], dtype=torch.int64)
        }
        
        if self.transform:
            image, target = self.transform(image, target)
            
        return image, target

# DETR Transform
class DetrTransform:
    """Resize / normalize / augment for DETR."""
    def __init__(self, size=IMAGE_SIZE, is_train=True):
        self.size = size
        self.is_train = is_train
        self.resize = T.Resize(size)
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if is_train:
            self.jitter = T.ColorJitter(0.2, 0.2, 0.2)
            self.flip = T.RandomHorizontalFlip(0.5)
        else:
            self.jitter = self.flip = None

    def __call__(self, img, tgt):
        boxes, labels = tgt["boxes"], tgt["labels"]
        img = self.resize(img)
        if self.is_train:
            img = self.jitter(img)
            if random.random() < 0.5:
                img = F.hflip(img)
                if boxes.numel() > 0:
                    boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        img = F.to_tensor(img)
        img = self.norm(img)
        tgt["boxes"] = boxes
        tgt["labels"] = labels
        tgt["size"] = torch.tensor([self.size[0], self.size[1]])
        return img, tgt

# Collate function
def collate_fn(batch):
    imgs, targs = zip(*batch)
    return torch.stack(imgs), list(targs)

# Create data loaders
def make_loaders(tr, vl, ts, bs=BATCH_SIZE, nw=NUM_WORKERS):
    tr_ds = AUAIRDataset(tr, IMAGES_DIR, transform=DetrTransform(IMAGE_SIZE, True))
    vl_ds = AUAIRDataset(vl, IMAGES_DIR, transform=DetrTransform(IMAGE_SIZE, False))
    ts_ds = AUAIRDataset(ts, IMAGES_DIR, transform=DetrTransform(IMAGE_SIZE, False))
    
    print(f"Data split: {len(tr_ds)} train, {len(vl_ds)} validation, {len(ts_ds)} test")
    
    return (
        DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=nw, pin_memory=True),
        DataLoader(vl_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=nw, pin_memory=True),
        DataLoader(ts_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=nw, pin_memory=True),
    )

# Build model
def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """Load or create DETR."""
    if pretrained:
        m = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", num_labels=num_classes, ignore_mismatched_sizes=True
        )
        # adjust head
        m.config.num_labels = num_classes
        m.class_labels_classifier = nn.Linear(256, num_classes+1)
    else:
        cfg = DetrConfig(num_labels=num_classes, num_queries=100)
        m = DetrForObjectDetection(cfg)
    return m

# DETR Object Detector
class DetrObjectDetector(nn.Module):
    """Wrapper for DETR training / inference."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.model = build_model(num_classes, pretrained)
        
    def forward(self, pixel_values, pixel_mask=None, labels=None):
        if labels is not None:
            tl = []
            for t in labels:
                tl.append({"class_labels": t["labels"], "boxes": t["boxes"]})
            return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=tl)
        else:
            return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            
    def predict(self, pixel_values, threshold=CONF_THRESH):
        out = self.model(pixel_values=pixel_values)
        logits = out.logits
        boxes = out.pred_boxes
        probs = torch.softmax(logits, dim=-1)
        scores, labels = torch.max(probs[..., 1:], dim=-1)
        labels += 1  # Adjust for background class
        
        batch_size = pixel_values.shape[0]
        preds = []
        
        for i in range(batch_size):
            m = scores[i] > threshold
            preds.append({
                "scores": scores[i][m],
                "labels": labels[i][m],
                "boxes": boxes[i][m]
            })
            
        return preds

# Training function
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total = {"loss": 0, "loss_ce": 0, "loss_bbox": 0, "loss_giou": 0}
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        targets = [
            {key: val.to(device) for key, val in t.items()}
            for t in targets
        ]
        
        # Forward pass
        outputs = model(pixel_values=images, labels=targets)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if CLIP_MAX_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
            
        optimizer.step()
        
        # Update metrics
        total["loss"] += loss.item()
        for k, v in loss_dict.items():
            if k not in total:
                total[k] = 0
            total[k] += v.item()
            
        # Update progress bar
        postfix = {
            "loss": total["loss"] / (batch_idx + 1),
            "CE": total.get("loss_ce", 0) / (batch_idx + 1),
            "BB": total.get("loss_bbox", 0) / (batch_idx + 1),
            "GIOU": total.get("loss_giou", 0) / (batch_idx + 1),
        }
        progress_bar.set_postfix(postfix)
        
        # Log to wandb
        global_step = epoch * len(loader) + batch_idx
        log_data = {
            "train/loss": loss.item(),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch,
            "global_step": global_step
        }
        
        for k, v in loss_dict.items():
            log_data[f"train/{k}"] = v.item()
            
        wandb.log(log_data)
        
    # Calculate average losses
    avg_loss = total["loss"] / len(loader)
    avg_loss_dict = {k: v / len(loader) for k, v in total.items()}
    
    return avg_loss, avg_loss_dict

# Evaluation function
def validate(model, loader, device, epoch):
    """Evaluate the model on the validation set."""
    model.eval()
    total = {"loss": 0, "loss_ce": 0, "loss_bbox": 0, "loss_giou": 0}
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device)
            targets = [
                {key: val.to(device) for key, val in t.items()}
                for t in targets
            ]
            
            # Forward pass
            outputs = model(pixel_values=images, labels=targets)
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            
            # Update metrics
            total["loss"] += loss.item()
            for k, v in loss_dict.items():
                if k not in total:
                    total[k] = 0
                total[k] += v.item()
                
            # Update progress bar
            postfix = {
                "loss": total["loss"] / (batch_idx + 1),
                "CE": total.get("loss_ce", 0) / (batch_idx + 1),
                "BB": total.get("loss_bbox", 0) / (batch_idx + 1),
                "GIOU": total.get("loss_giou", 0) / (batch_idx + 1),
            }
            progress_bar.set_postfix(postfix)
            
    # Calculate average losses
    avg_loss = total["loss"] / len(loader)
    avg_loss_dict = {k: v / len(loader) for k, v in total.items()}
    
    # Log to wandb
    log_data = {
        "val/loss": avg_loss,
        "epoch": epoch
    }
    
    for k, v in avg_loss_dict.items():
        log_data[f"val/{k}"] = v
        
    wandb.log(log_data)
    
    return avg_loss, avg_loss_dict

# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, loss):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"detr_auair_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

# Load checkpoint
def load_checkpoint(model, optimizer=None, scheduler=None, path=None):
    """Load model checkpoint."""
    if path is None:
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("detr_auair_epoch_")]
        if not checkpoints:
            print("No checkpoints found.")
            return model, optimizer, scheduler, 0, float("inf")
        
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    
    return model, optimizer, scheduler, epoch, loss

# Compute IoU
def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

# Compute AP
def compute_ap(precisions, recalls):
    """Compute Average Precision."""
    p, r = np.array(precisions), np.array(recalls)
    idx = np.argsort(r)
    r, p = r[idx], p[idx]
    
    r = np.concatenate(([0.], r, [1.]))
    p = np.concatenate(([0.], p, [0.]))
    
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
        
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.any(r >= t):
            ap += np.max(p[r >= t]) / 11
            
    return ap

# Evaluate model
def evaluate(model, loader, device, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """Evaluate model on test set."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, batch_targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            batch_preds = model.predict(images, threshold=conf_thresh)
            
            predictions.extend(batch_preds)
            targets.extend(batch_targets)
    
    # Compute AP for each class
    class_aps = []
    results = {}
    
    for class_id in range(NUM_CLASSES):
        precisions = []
        recalls = []
        
        for threshold in np.arange(0, 1, 0.1):
            tp = fp = fn = 0
            
            for pred, target in zip(predictions, targets):
                # Get predictions for this class
                pred_indices = (pred["labels"] == class_id).nonzero().flatten()
                pred_boxes = pred["boxes"][pred_indices]
                pred_scores = pred["scores"][pred_indices]
                
                # Get ground truth for this class
                target_indices = (target["labels"] == class_id).nonzero().flatten()
                target_boxes = target["boxes"][target_indices]
                
                # Filter predictions by threshold
                keep = pred_scores >= threshold
                pred_boxes = pred_boxes[keep]
                
                # Match predictions to ground truth
                matches = set()
                
                for pred_box in pred_boxes:
                    best_iou = 0
                    best_idx = -1
                    
                    for j, gt_box in enumerate(target_boxes):
                        iou = compute_iou(pred_box.cpu().numpy(), gt_box.cpu().numpy())
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = j
                            
                    if best_iou >= iou_thresh and best_idx not in matches:
                        tp += 1
                        matches.add(best_idx)
                    else:
                        fp += 1
                        
                fn += len(target_indices) - len(matches)
                
            # Compute precision and recall
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            
        # Compute AP
        ap = compute_ap(precisions, recalls)
        class_aps.append(ap)
        
        # Store results
        results[CLASS_NAMES[class_id]] = {
            "AP": ap,
            "precisions": precisions,
            "recalls": recalls
        }
    
    # Compute mAP
    mAP = np.mean(class_aps)
    results["mAP"] = mAP
    
    return mAP, class_aps, results

# Main function
def main(pretrained=True, num_epochs=NUM_EPOCHS, resume=False, checkpoint_path=None):
    """Main training function."""
    print("Loading annotations...")
    ann = load_annotations()
    
    print("Creating data splits...")
    train_anns, val_anns, test_anns = create_splits(ann)
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = make_loaders(train_anns, val_anns, test_anns)
    
    print("Creating model...")
    model = DetrObjectDetector(pretrained=pretrained).to(device)
    
    if pretrained:
        print("Loaded pretrained model: facebook/detr-resnet-50")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=LR_SCHED_TMAX,
        eta_min=LR_SCHED_ETA_MIN
    )
    
    # Initialize wandb
    init_wandb()
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float("inf")
    
    if resume:
        model, optimizer, scheduler, start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path
        )
        start_epoch += 1
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        # Validate
        val_loss, val_loss_dict = validate(model, val_loader, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_FREQ == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss)
            
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    mAP, class_aps, results = evaluate(model, test_loader, device)
    
    print(f"mAP: {mAP:.4f}")
    for i, ap in enumerate(class_aps):
        print(f"{CLASS_NAMES[i]}: {ap:.4f}")
    
    # Log results to wandb
    wandb.log({
        "test/mAP": mAP,
        **{f"test/AP_{CLASS_NAMES[i]}": ap for i, ap in enumerate(class_aps)}
    })
    
    # Finish wandb run
    wandb.finish()
    
    return model, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DETR on AU-AIR dataset")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    
    args = parser.parse_args()
    
    main(
        pretrained=args.pretrained,
        num_epochs=args.epochs,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )
