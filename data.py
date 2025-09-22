import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

# --------------------------
# Config
# --------------------------
IMG_SIZE = 224

TRAIN_DIR = "/media/jag/volD2/imagent-100/train"
VAL_DIR = "/media/jag/volD2/imagent-100/val"
FAKE_DIR = "./fooling_images"

BATCH_SIZE = 64

# --------------------------
# Data Transforms
# --------------------------
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# --------------------------
# Custom Dataset for Class Range Filtering
# --------------------------
class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, start_class=0, end_class=None, preserve_indices=False):
        super().__init__(root, transform=transform)
        
        if end_class is None:
            end_class = len(self.classes) - 1
            
        # Filter samples based on class range
        self.filtered_samples = []
        self.filtered_targets = []
        self.class_mapping = {}
        self.preserve_indices = preserve_indices
        
        # Create mapping from original class indices to new indices
        selected_classes = list(range(start_class, min(end_class + 1, len(self.classes))))
        
        if preserve_indices:
            # Keep original class indices
            for orig_idx in selected_classes:
                self.class_mapping[orig_idx] = orig_idx
        else:
            # Remap to 0-based indices
            for new_idx, orig_idx in enumerate(selected_classes):
                self.class_mapping[orig_idx] = new_idx
        
        # Filter samples
        for path, target in self.samples:
            if start_class <= target <= end_class:
                self.filtered_samples.append((path, self.class_mapping[target]))
                self.filtered_targets.append(self.class_mapping[target])
        
        # Update class information
        self.filtered_classes = [self.classes[i] for i in selected_classes]
        self.samples = self.filtered_samples
        self.targets = self.filtered_targets
        
        if not preserve_indices:
            self.classes = self.filtered_classes
    
    def __len__(self):
        return len(self.filtered_samples)


def get_loaders(batch_size=BATCH_SIZE, num_workers=4, start_range=0, end_range=99, data_ratio=1, preserve_indices=False):
    train_transform, val_transform = get_transforms()
    
    print(f"Loading data for classes {start_range} to {end_range}...")

    # Create filtered datasets
    train_dataset = FilteredImageFolder(
        TRAIN_DIR, 
        transform=train_transform, 
        start_class=start_range, 
        end_class=end_range,
        preserve_indices=preserve_indices
    )
    val_dataset = FilteredImageFolder(
        VAL_DIR, 
        transform=val_transform, 
        start_class=start_range, 
        end_class=end_range,
        preserve_indices=preserve_indices
    )
    
    # Apply data ratio filtering
    train_size = int(len(train_dataset) * data_ratio)
    val_size = int(len(val_dataset) * data_ratio)
    
    # Create subset datasets
    train_indices = torch.randperm(len(train_dataset))[:train_size]
    val_indices = torch.randperm(len(val_dataset))[:val_size]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print percentage information
    print(f"Using {data_ratio*100:.1f}% of dataset")
    print(f"Train samples: {train_size} ({train_size/len(train_dataset)*100:.1f}%)")
    print(f"Val samples: {val_size} ({val_size/len(val_dataset)*100:.1f}%)")
    
    # Get class names
    classes = train_dataset.classes
    
    return train_loader, val_loader, classes


