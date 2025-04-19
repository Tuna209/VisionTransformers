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
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm
import wandb
from datetime import datetime
from transformers import DetrForObjectDetection, DetrImageProcessor

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
    def __init__(self, anns, img_dir, processor):
        self.anns = anns
        self.img_dir = img_dir
        self.processor = processor

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
            x_min = b["left"]
            y_min = b["top"]
            width = b["width"]
            height = b["height"]
            class_idx = b["class"]

            x_max = x_min + width
            y_max = y_min + height

            # Ensure coordinates are valid
            if x_max <= x_min or y_max <= y_min or width <= 0 or height <= 0:
                continue

            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_idx)

        # If no valid boxes, return a random image
        if len(boxes) == 0:
            random_idx = random.randint(0, len(self.anns) - 1)
            while random_idx == i:
                random_idx = random.randint(0, len(self.anns) - 1)
            return self.__getitem__(random_idx)

        # Convert to COCO format for the processor
        coco_annotations = {
            "image_id": i,
            "annotations": [
                {
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # COCO format: [x, y, width, height]
                    "category_id": label,
                    "id": j,
                    "area": (box[2] - box[0]) * (box[3] - box[1]),
                    "iscrowd": 0
                } for j, (box, label) in enumerate(zip(boxes, labels))
            ]
        }

        # Process with DETR processor
        encoding = self.processor(images=image, annotations=coco_annotations, return_tensors="pt")

        # Remove batch dimension
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1:
                encoding[k] = v[0]

        # Add the target tensors directly
        encoding["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        encoding["labels"] = torch.tensor(labels, dtype=torch.int64)
        encoding["image_id"] = torch.tensor(i)
        encoding["orig_size"] = torch.tensor([h, w])
        encoding["orig_image"] = image

        return encoding

# Collate function for data loader
def collate_fn(batch):
    """Custom collate function for DETR data loader."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])

    # Prepare labels in the format expected by DETR
    labels = []
    for item in batch:
        # Create a target dict with the required keys
        target = {
            "boxes": item["boxes"],
            "class_labels": item["labels"],  # Renamed to class_labels for DETR
            "image_id": item["image_id"],
        }
        labels.append(target)

    orig_sizes = [item["orig_size"] for item in batch]
    orig_images = [item["orig_image"] for item in batch]

    batch = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
        "orig_sizes": orig_sizes,
        "orig_images": orig_images
    }

    return batch

# Create data loaders
def create_data_loaders(processor):
    """Create data loaders for training, validation, and testing."""
    # Load annotations
    ann = load_annotations()

    # Create splits
    train_anns, val_anns, test_anns = create_splits(ann)

    # Create datasets
    train_dataset = AUAIRDataset(train_anns, IMAGES_DIR, processor)
    val_dataset = AUAIRDataset(val_anns, IMAGES_DIR, processor)
    test_dataset = AUAIRDataset(test_anns, IMAGES_DIR, processor)

    print(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader

# Training function
def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    loss_dict_sum = {}

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in label.items()} for label in batch["labels"]]

        # Forward pass
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
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
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_sum:
                loss_dict_sum[k] = 0
            loss_dict_sum[k] += v.item()

        # Update progress bar
        postfix = {"loss": loss.item()}
        for k, v in loss_dict.items():
            postfix[k] = v.item()
        progress_bar.set_postfix(postfix)

        # Log to wandb with step information
        global_step = epoch * len(dataloader) + batch_idx
        log_data = {
            "train/loss": loss.item(),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch,
            "global_step": global_step
        }

        # Add all loss components
        for k, v in loss_dict.items():
            log_data[f"train/{k}"] = v.item()

        wandb.log(log_data)

    # Update scheduler
    scheduler.step()

    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}

    return avg_loss, avg_loss_dict

# Evaluation function
def evaluate(model, dataloader, device, epoch=0):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    loss_dict_sum = {}

    progress_bar = tqdm(dataloader, desc="Evaluation")

    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in label.items()} for label in batch["labels"]]

            # Forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            loss_dict = outputs.loss_dict

            # Update metrics
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_dict_sum:
                    loss_dict_sum[k] = 0
                loss_dict_sum[k] += v.item()

            # Update progress bar
            postfix = {"loss": loss.item()}
            for k, v in loss_dict.items():
                postfix[k] = v.item()
            progress_bar.set_postfix(postfix)

    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}

    # Log to wandb with epoch information
    log_data = {
        "val/loss": avg_loss,
        "epoch": epoch
    }

    # Add all loss components
    for k, v in avg_loss_dict.items():
        log_data[f"val/{k}"] = v

    wandb.log(log_data)

    return avg_loss, avg_loss_dict

# Visualization function
def visualize_results(model, dataloader, processor, device, num_images=10, confidence_threshold=0.5, output_dir=None, use_wandb=True, epoch=None):
    """Visualize model predictions on test images."""
    model.eval()
    images_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if images_processed >= num_images:
                break

            # Move data to device
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            orig_images = batch["orig_images"]
            orig_sizes = batch["orig_sizes"]

            # Forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # Process each image in the batch
            for i in range(len(orig_images)):
                if images_processed >= num_images:
                    break

                # Get predictions
                target_sizes = torch.tensor([orig_sizes[i].tolist()])
                results = processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=confidence_threshold
                )[0]

                # Get ground truth
                gt_boxes = batch["labels"][i]["boxes"].cpu()
                gt_labels = batch["labels"][i]["class_labels"].cpu()

                # Create visualization
                image = orig_images[i]
                pred_image = image.copy()
                gt_image = image.copy()

                # Draw predictions
                draw_pred = ImageDraw.Draw(pred_image)
                for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                    box = box.tolist()
                    x_min, y_min, x_max, y_max = box

                    # Ensure coordinates are valid
                    if x_max > x_min and y_max > y_min:
                        # Draw rectangle
                        draw_pred.rectangle(
                            [(x_min, y_min), (x_max, y_max)],
                            outline='red',
                            width=2
                        )

                        # Draw label
                        class_name = CLASS_NAMES[label]
                        draw_pred.text(
                            (x_min, y_min - 10),
                            f"{class_name}: {score:.2f}",
                            fill='red'
                        )

                # Draw ground truth
                draw_gt = ImageDraw.Draw(gt_image)
                for box, label in zip(gt_boxes, gt_labels):
                    box = box.tolist()
                    x_min, y_min, x_max, y_max = box

                    # Ensure coordinates are valid
                    if x_max > x_min and y_max > y_min:
                        # Draw rectangle
                        draw_gt.rectangle(
                            [(x_min, y_min), (x_max, y_max)],
                            outline='green',
                            width=2
                        )

                        # Draw label
                        class_name = CLASS_NAMES[label]
                        draw_gt.text(
                            (x_min, y_min - 10),
                            class_name,
                            fill='green'
                        )

                # Create side-by-side comparison
                comparison = Image.new('RGB', (image.width * 2, image.height))
                comparison.paste(pred_image, (0, 0))
                comparison.paste(gt_image, (image.width, 0))

                # Save image
                if output_dir:
                    comparison.save(os.path.join(output_dir, f"comparison_{images_processed}.jpg"))

                # Log to wandb
                if use_wandb:
                    log_data = {
                        f"visualization/image_{images_processed}": wandb.Image(
                            comparison,
                            caption=f"Left: Predictions, Right: Ground Truth"
                        )
                    }

                    # Add epoch information if provided
                    if epoch is not None:
                        log_data["epoch"] = epoch

                    wandb.log(log_data)

                images_processed += 1

# Main function
def main(pretrained=True, skip_train=False, skip_eval=False, skip_viz=False, no_wandb=False):
    """Main function to run the training pipeline."""
    # Initialize wandb
    if not no_wandb:
        init_wandb()

    # Initialize model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size=IMAGE_SIZE)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(processor)

    # Create model
    print("Creating model...")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    if pretrained:
        print("Loaded pretrained model: facebook/detr-resnet-50")

    model.to(device)

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

    # Training loop
    if not skip_train:
        print(f"Starting training for {NUM_EPOCHS} epochs...")
        for epoch in range(NUM_EPOCHS):
            # Train
            train_loss, train_loss_dict = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch
            )

            # Evaluate
            if not skip_eval:
                val_loss, val_loss_dict = evaluate(model, val_loader, device, epoch)

                print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

            # Save checkpoint and visualize intermediate results
            if (epoch + 1) % SAVE_FREQ == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"detr_auair_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss if not skip_eval else None
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

                # Log sample visualizations to track progress
                if not skip_viz and not no_wandb:
                    print(f"Generating visualizations for epoch {epoch}...")
                    visualize_results(
                        model,
                        val_loader,  # Use validation set for visualizations
                        processor,
                        device,
                        num_images=2,  # Just a few images to track progress
                        confidence_threshold=CONF_THRESH,
                        output_dir=None,
                        use_wandb=True,
                        epoch=epoch
                    )

    # Evaluate on test set
    if not skip_eval:
        print("Evaluating on test set...")
        test_loss, test_loss_dict = evaluate(model, test_loader, device, NUM_EPOCHS)
        print(f"Test Loss: {test_loss:.4f}")

        if not no_wandb:
            log_data = {"test/loss": test_loss}
            for k, v in test_loss_dict.items():
                log_data[f"test/{k}"] = v
            wandb.log(log_data)

    # Visualize results
    if not skip_viz:
        print("Visualizing results...")
        visualize_results(
            model,
            test_loader,
            processor,
            device,
            num_images=NUM_VIZ_IMAGES,
            confidence_threshold=CONF_THRESH,
            output_dir=VIZ_OUTPUT_DIR,
            use_wandb=not no_wandb,
            epoch=NUM_EPOCHS  # Final epoch
        )

    # Finish wandb run
    if not no_wandb:
        wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DETR Fine-tuning on AU-AIR Dataset")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--skip_train", action="store_true", help="Skip training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip_viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")

    args = parser.parse_args()

    # Update global variables based on arguments
    NUM_EPOCHS = args.num_epochs

    main(
        pretrained=args.pretrained,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        skip_viz=args.skip_viz,
        no_wandb=args.no_wandb
    )
