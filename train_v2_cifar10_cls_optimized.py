# train_v2_cifar10_cls_optimized.py
# Optimized training script for transformer-level speed
import torch
from torch import nn
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os, math, time, csv
import argparse
import glob
import random
import numpy as np

from models.bdh import BDHConfig
from models.vision_bdh_v2_cls_optimized import VisionBDHv2CLSOptimized

def fix_seed(seed: int = 42):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Some dataloader behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA >=10.2

    print(f"[Seed fixed to {seed}]")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Creates a learning rate schedule with linear warmup followed by cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def main(args):
    # --- Configuration ---
    EPOCHS = 50
    BATCH_SIZE = 128  # Increased from 64 for better GPU utilization
    GRAD_ACCUM_STEPS = 2  # Effective batch size = 256
    INITIAL_LR = 2e-4
    WARMUP_STEPS = 1000
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints_v2_cifar10_cls_optimized"
    LOG_FILE = os.path.join(CHECKPOINT_DIR, "metrics_v2_cifar10.csv")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # GPU performance tweaks - more aggressive
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    # Enable cudnn benchmark for faster convolutions (set to True for speed)
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    print("=" * 70)
    print("     Training Vision-BDH v2 OPTIMIZED")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print("=" * 70)

    # BDH config
    config = BDHConfig(
        n_layer=6,
        n_embd=192,
        n_head=6,
        vocab_size=256,
        mlp_internal_dim_multiplier=32,
        dropout=0.1
    )

    model = VisionBDHv2CLSOptimized(
        bdh_config=config,
        img_size=32,
        patch_size=4,
        num_classes=10,
        use_softmax_attn=False
    )

    # Convert patch_embed to channels_last for better conv performance
    model.patch_embed = model.patch_embed.to(memory_format=torch.channels_last)

    # --- Model Compilation with inductor (more aggressive) ---
    print("\nCompiling Vision-BDH v2 with inductor backend...")
    try:
        # Use inductor backend for best performance (PyTorch 2.0+)
        model = torch.compile(model, mode="max-autotune", backend="inductor")
        print(f"✓ Model compiled successfully with inductor backend.")
    except Exception as e:
        print(f"⚠️ Warning: Inductor compilation failed, trying default...")
        try:
            model = torch.compile(model)
            print(f"✓ Model compiled with default backend.")
        except Exception as e2:
            print(f"⚠️ Warning: Compilation failed, continuing without it. Error: {e2}")

    model.to(DEVICE)

    # Use fused AdamW for better performance
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.05, fused=True)

    # --- Resume Logic ---
    start_epoch = 0
    if args.resume:
        list_of_files = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pth'))
        if not list_of_files:
            print("No checkpoint found to resume from.")
        else:
            latest_checkpoint_path = max(list_of_files, key=os.path.getctime)
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created with {num_params / 1e6:.2f}M trainable parameters.\n")

    # --- Data Preparation with optimizations ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = CIFAR10(root="/home/sunghyun/bdh-vision/data", train=True, download=True, transform=train_transform)
    train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = CIFAR10(root="/home/sunghyun/bdh-vision/data", train=False, download=True, transform=val_test_transform)

    # Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,  # Increased from 2
        pin_memory=True,
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True  # Keep workers alive
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE * 2,  # Larger batch for validation
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE * 2, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    print(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # --- Scheduler ---
    num_training_steps = EPOCHS * len(train_loader) // GRAD_ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps
    )

    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # --- Training ---
    print("\n" + "=" * 70)
    print("     Starting Training Loop (OPTIMIZED)")
    print("=" * 70 + "\n")

    # Initialize metrics log
    if start_epoch == 0:
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_accuracy", "epoch_time_sec", "learning_rate"])

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        
        # Set zero_grad before loop (more efficient than calling each iteration)
        optimizer.zero_grad(set_to_none=True)

        for i, (images, labels) in enumerate(train_loader):
            # Use channels_last for input as well
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss = loss / GRAD_ACCUM_STEPS  # Scale loss for gradient accumulation
            
            # Backward pass
            scaler.scale(loss).backward()

            # Update weights every GRAD_ACCUM_STEPS
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Zero gradients for next accumulation
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUM_STEPS

            if (i + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, "
                      f"Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}, LR: {current_lr:.6f}")

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(DEVICE, non_blocking=True)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(images)
                
                preds = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start_time

        print("-" * 70)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print("-" * 70)

        # Save metrics
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, val_accuracy, epoch_time, scheduler.get_last_lr()[0]])

        # --- Save Checkpoint (less frequently to save time) ---
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}\n")

            # Remove old checkpoints to save space (keep every 5th and last)
            for old_epoch in range(max(0, epoch - 10), epoch):
                if old_epoch % 5 != 0:
                    old_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{old_epoch}.pth')
                    if os.path.exists(old_checkpoint_path):
                        os.remove(old_checkpoint_path)

    # --- Final Evaluation ---
    print("\n" + "=" * 70)
    print("     Final Evaluation on Test Set (Best Checkpoint)")
    print("=" * 70)

    best_acc = 0
    best_path = ""
    for ckpt_path in glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")):
        ckpt = torch.load(ckpt_path)
        if 'val_accuracy' in ckpt and ckpt['val_accuracy'] > best_acc:
            best_acc = ckpt['val_accuracy']
            best_path = ckpt_path

    print(f"Loading best model from: {best_path} (val_acc={best_acc:.2f}%)")
    if best_path:
        ckpt = torch.load(best_path)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
            
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    final_model_path = os.path.join(CHECKPOINT_DIR, "final_model_best_v2.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisionBDH v2 (Optimized) on CIFAR-10")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic algorithms (slower but reproducible)")
    args = parser.parse_args()
    
    if args.deterministic:
        fix_seed(args.seed)
    else:
        # Still set seed but allow non-deterministic ops for speed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[Seed set to {args.seed} (non-deterministic mode for speed)]")
    
    main(args)
