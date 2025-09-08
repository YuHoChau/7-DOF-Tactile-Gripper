import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Suppress xFormers warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

# ---------- Configuration ----------
DEFAULT_DATASET_DIR = "/new_dataset"
DEFAULT_BASE_IMAGE_PATH = "/base_clear.png"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
IMG_SIZE = 224
RANDOM_SEED = 42

# Set random seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- Image Encoder Functions ----------
def create_resnet_encoder():
    """Create frozen ResNet-18 encoder"""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in resnet.parameters():
        param.requires_grad = False  # Freeze ResNet-18
    
    encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
    feature_dim = 512  # ResNet-18 feature dimension
    return encoder, feature_dim


def get_image_encoder(encoder_type='resnet'):
    """Get image encoder based on type"""
    if encoder_type == 'resnet':
        return create_resnet_encoder()
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

# ---------- Data Preprocessing ----------
# Base image preprocessing (only resize, no normalization)
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Final image preprocessing (for difference image normalization)
image_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

def load_base_image(base_image_path):
    """Load base image for difference calculation"""
    print(f"Loading base image: {base_image_path}")
    try:
        base_image_pil = Image.open(base_image_path).convert("RGB")
        base_image_tensor = base_transform(base_image_pil)  # [3, IMG_SIZE, IMG_SIZE]
        print(f"Base image loaded successfully, shape: {base_image_tensor.shape}")
        return base_image_tensor
    except Exception as e:
        print(f"Cannot load base image {base_image_path}: {e}")
        # Create a zero image as fallback
        base_image_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        print("Using zero image as base image")
        return base_image_tensor

def get_diff_image(current_image_pil, base_image_tensor):
    """
    Calculate the difference between current image and base image
    Args:
        current_image_pil: PIL image
        base_image_tensor: Base image tensor
    Returns:
        diff_image_tensor: Difference image tensor, already normalized with ImageNet
    """
    # 1. Convert current image to tensor (only resize, no normalization)
    current_image_tensor = base_transform(current_image_pil)  # [3, IMG_SIZE, IMG_SIZE]
    
    # 2. Calculate difference image
    diff_image = current_image_tensor - base_image_tensor
    
    # 3. Scale difference range from [-1, 1] to [0, 1]
    # Because original image range is [0, 1], difference range is [-1, 1]
    diff_image = (diff_image + 1.0) / 2.0
    
    # 4. Ensure within [0, 1] range
    diff_image = torch.clamp(diff_image, 0.0, 1.0)
    
    # 5. Apply ImageNet normalization
    diff_image = image_transform(diff_image)
    
    return diff_image

def normalize_tactile_data(tactile_data):
    """Normalize tactile data: divide by 255"""
    return np.array(tactile_data) / 255.0

def normalize_motor_joints(joints):
    """Normalize motor joints: (x - 180.0) / (220.0 - 180.0)"""
    joints = np.array(joints)
    normalized_joints = (joints - 180.0) / (220.0 - 180.0)
    return normalized_joints

# ---------- Cross Attention Module ----------
class CrossAttention(nn.Module):
    """Unified Cross Attention for 2+ modalities - supports any number of modalities"""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # Fusion layer will be dynamically created based on input modalities
        self.fusion = None
        
    def _create_fusion_layer(self, num_modalities: int, dim: int):
        """Dynamically create fusion layer based on number of modalities"""
        if self.fusion is None:
            self.fusion = nn.Linear(num_modalities * dim, dim)
            # Move to same device as other parameters
            if hasattr(self, 'to_q'):
                self.fusion = self.fusion.to(next(self.parameters()).device)
        
    def forward(self, *modalities):
        """
        Unified cross attention for any number of modalities (≥2)
        Args:
            *modalities: Variable number of modality tensors [B, D]
        Returns:
            fused_features: [B, D]
            attention_weights: Dict with attention weights for each modality
        """
        num_modalities = len(modalities)
        if num_modalities < 2:
            raise ValueError("At least 2 modalities required for cross attention")
        
        # Create fusion layer if not exists
        self._create_fusion_layer(num_modalities, modalities[0].size(-1))
        
        # 1. Generate queries for all modalities
        qs = torch.stack([self.to_q(mod) for mod in modalities], dim=1)  # [B, N, D]
        
        # 2. Generate cross KV pairs
        # For each modality i, its KV comes from all OTHER modalities
        kv_inputs = []
        for i in range(num_modalities):
            # For modality i, use all other modalities as KV
            other_modalities = [modalities[j] for j in range(num_modalities) if j != i]
            kv_inputs.extend(other_modalities)
        
        kv_inputs = torch.stack(kv_inputs, dim=1)  # [B, N*(N-1), D]
        
        # Generate KV pairs and reshape
        kvs = self.to_kv(kv_inputs).chunk(num_modalities, dim=1)  # Split into N groups
        kvs = [kv.view(kv.size(0), num_modalities-1, 2, -1) for kv in kvs]  # [B, N-1, 2, D]
        
        # 3. Compute attention for each modality
        outputs = []
        attn_weights = []
        
        for i in range(num_modalities):
            q = qs[:, i].unsqueeze(1)  # [B, 1, D]
            k, v = kvs[i].unbind(2)   # k/v: [B, N-1, D]
            
            # Attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, 1, N-1]
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v)[:, 0, :]  # [B, D]
            
            outputs.append(out)
            attn_weights.append(attn.squeeze(1))  # [B, N-1] or [B] for 2 modalities
        
        # 4. Fuse outputs
        fused = self.fusion(torch.cat(outputs, dim=-1))
        
        # 5. Create attention weights dictionary
        attn_dict = {
            f'attn_{i}': attn_weights[i] for i in range(num_modalities)
        }
        
        return fused, attn_dict

class MultiModalCrossAttention(nn.Module):
    """Multi-modal Cross Attention fusion module - now using unified attention"""
    def __init__(self, modalities, image_dim=512, tactile_dim=64, motor_dim=3, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        self.modalities = modalities
        self.hidden_dim = hidden_dim
        
        # Project each modality feature to the same dimension
        if 'tactile_image' in modalities:
            self.image_proj = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        if 'tactile_array' in modalities:
            self.tactile_proj = nn.Sequential(
                nn.Linear(tactile_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        if 'proprioception' in modalities:
            self.motor_proj = nn.Sequential(
                nn.Linear(motor_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.cross_attention = CrossAttention(dim=hidden_dim, dropout=dropout)
        
        # Final feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        print(f"Multi-modal Cross Attention mechanism created:")
        print(f"  - Modalities: {modalities}")
        print(f"  - Projection dimension: {hidden_dim}")
    
    def forward(self, **features):
        projected_features = []
        
        if 'tactile_image' in self.modalities and 'image_features' in features:
            img_proj = self.image_proj(features['image_features'])
            projected_features.append(img_proj)
        
        if 'tactile_array' in self.modalities and 'tactile_features' in features:
            tac_proj = self.tactile_proj(features['tactile_features'])
            projected_features.append(tac_proj)
        
        if 'proprioception' in self.modalities and 'motor_features' in features:
            mot_proj = self.motor_proj(features['motor_features'])
            projected_features.append(mot_proj)
        
        fused_features, attention_weights = self.cross_attention(*projected_features)
        
        enhanced_features = self.feature_enhance(fused_features)
        
        final_features = enhanced_features + fused_features
        
        return final_features, attention_weights

# ---------- Single Modality Classifiers ----------

class SingleImageClassifier(nn.Module):
    """Single modality classifier for image only"""
    def __init__(self, num_classes):
        super().__init__()
        
        # Image encoder - always ResNet
        self.image_encoder, image_feature_dim = get_image_encoder('resnet')
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"Single Image Classifier created:")
        print(f"  - Image encoder: ResNet")
        print(f"  - Image feature dimension: {image_feature_dim}")
        print(f"  - Number of classes: {num_classes}")
    
    def forward(self, image):
        # Image feature extraction
        # ResNet returns [B, feature_dim, 1, 1], need to flatten
        image_features = self.image_encoder(image)  # [B, feature_dim, 1, 1]
        image_features = image_features.view(image_features.size(0), -1)  # [B, feature_dim]
        
        # Classification
        output = self.classifier(image_features)
        
        return output, None  # No attention weights for single modality

class SingleTactileClassifier(nn.Module):
    """Single modality classifier for tactile only"""
    def __init__(self, num_classes, tactile_dim=384):
        super().__init__()
        
        # Use MLP encoder (same as in multi-modal)
        self.tactile_encoder = nn.Sequential(
            nn.Linear(tactile_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)
        )
        input_dim = 64
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"Single Tactile Classifier created:")
        print(f"  - Encoder type: MLP encoder")
        print(f"  - Input dimension: {tactile_dim}")
        print(f"  - Feature dimension: {input_dim}")
        print(f"  - Number of classes: {num_classes}")
    
    def forward(self, tactile):
        # Tactile feature extraction
        tactile_features = self.tactile_encoder(tactile)
        
        # Classification
        output = self.classifier(tactile_features)
        
        return output, None  # No attention weights for single modality

class SingleMotorClassifier(nn.Module):
    """Single modality classifier for motor/proprioception only"""
    def __init__(self, num_classes, motor_joints_dim=3):
        super().__init__()
        
        # Simple 3-layer MLP for motor joints
        self.motor_encoder = nn.Sequential(
            nn.Linear(motor_joints_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        print(f"Single Motor Classifier created:")
        print(f"  - Input dimension: {motor_joints_dim}")
        print(f"  - Encoder: 3-layer MLP (64->32->16)")
        print(f"  - Number of classes: {num_classes}")
    
    def forward(self, motor_joints):
        # Motor feature extraction
        motor_features = self.motor_encoder(motor_joints)
        
        # Classification
        output = self.classifier(motor_features)
        
        return output, None  # No attention weights for single modality

# ---------- Fusion Model Definition ----------
class MultiModalFusionClassifier(nn.Module):
    def __init__(self, num_classes, modalities, tactile_dim=384, motor_joints_dim=3):
        super().__init__()
        
        self.modalities = modalities
        
        # Check if single modality
        if len(modalities) == 1:
            modality = modalities[0]
            if modality == 'tactile_image':
                self.single_classifier = SingleImageClassifier(num_classes)
            elif modality == 'tactile_array':
                self.single_classifier = SingleTactileClassifier(num_classes, tactile_dim)
            elif modality == 'proprioception':
                self.single_classifier = SingleMotorClassifier(num_classes, motor_joints_dim)
            self.is_single_modality = True
            return
        
        self.is_single_modality = False
        
        # Multi-modal setup (existing code)
        # 1. Image encoder - always ResNet
        if 'tactile_image' in modalities:
            self.image_encoder, image_feature_dim = get_image_encoder('resnet')
        else:
            image_feature_dim = 512  # Default
        
        # 2. Tactile encoder - improved MLP
        if 'tactile_array' in modalities:
            self.tactile_encoder = nn.Sequential(
                nn.Linear(tactile_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64)
            )
            tactile_feature_dim = 64
        else:
            tactile_feature_dim = 64
        
        # 3. Multi-modal Cross Attention mechanism
        attention_output_dim = 128
        self.attention_fusion = MultiModalCrossAttention(
            modalities=modalities,
            image_dim=image_feature_dim,
            tactile_dim=tactile_feature_dim,
            motor_dim=motor_joints_dim,
            hidden_dim=attention_output_dim
        )
        
        # 4. Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        print(f"Multi-modal fusion classifier created:")
        print(f"  - Modalities: {modalities}")
        print(f"  - Image encoder: ResNet")
        print(f"  - Image feature dimension: {image_feature_dim} (frozen, using difference image)")
        print(f"  - Tactile array feature dimension: {tactile_feature_dim} (improved MLP encoding)")
        print(f"  - Proprioception dimension: {motor_joints_dim}")
        print(f"  - Cross Attention output dimension: {attention_output_dim}")
        print(f"  - Number of classes: {num_classes}")

    def forward(self, **inputs):
        if self.is_single_modality:
            # Single modality forward
            modality = self.modalities[0]
            if modality == 'tactile_image':
                return self.single_classifier(inputs['image'])
            elif modality == 'tactile_array':
                return self.single_classifier(inputs['tactile'])
            elif modality == 'proprioception':
                return self.single_classifier(inputs['motor_joints'])
        
        # Multi-modal forward (existing code)
        features = {}
        
        # 1. Image feature extraction (input image is already a difference image)
        if 'tactile_image' in self.modalities and 'image' in inputs:
            # ResNet returns [B, 512, 1, 1], need to flatten
            image_features = self.image_encoder(inputs['image'])  # [B, 512, 1, 1]
            image_features = image_features.view(image_features.size(0), -1)  # [B, 512]
            
            features['image_features'] = image_features
        
        # 2. Tactile feature extraction
        if 'tactile_array' in self.modalities and 'tactile' in inputs:
            tactile_features = self.tactile_encoder(inputs['tactile'])  # [B, 64]
            features['tactile_features'] = tactile_features
        
        # 3. Proprioception features (direct use)
        if 'proprioception' in self.modalities and 'motor_joints' in inputs:
            features['motor_features'] = inputs['motor_joints']
        
        # 4. Multi-modal Cross Attention fusion
        fused_features, attention_weights = self.attention_fusion(**features)
        
        # 5. Classification
        output = self.classifier(fused_features)
        
        return output, attention_weights

# ---------- Fusion Dataset Definition ----------
class MultiModalDataset(Dataset):
    def __init__(self, sample_paths, labels, modalities, base_image_tensor):
        self.sample_paths = sample_paths
        self.labels = labels
        self.modalities = modalities
        self.base_image_tensor = base_image_tensor

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = Path(self.sample_paths[idx])
        label = self.labels[idx]
        
        result = {'label': torch.tensor(label, dtype=torch.long)}
        
        # 1. Load image if needed
        if 'tactile_image' in self.modalities:
            image_path = sample_path / "gelsight_raw_image.png"
            try:
                current_image_pil = Image.open(image_path).convert("RGB")
                # Calculate difference image
                diff_image = get_diff_image(current_image_pil, self.base_image_tensor)
                result['image'] = diff_image
            except Exception as e:
                print(f"Cannot load image {image_path}: {e}")
                # Create blank difference image as fallback
                diff_image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
                diff_image = image_transform(diff_image)
                result['image'] = diff_image
        
        # 2. Load tactile data if needed
        if 'tactile_array' in self.modalities:
            tactile_data = []
            finger_files = [
                sample_path / "finger1_tactile_array.txt",
                sample_path / "finger2_tactile_array.txt", 
                sample_path / "finger3_tactile_array.txt"
            ]
            
            for finger_file in finger_files:
                try:
                    with open(finger_file, 'r') as f:
                        lines = f.readlines()
                        finger_data = []
                        for line in lines:
                            line = line.strip()
                            if line:
                                finger_data.append(float(line))
                        tactile_data.extend(finger_data)
                except Exception as e:
                    print(f"Cannot load tactile file {finger_file}: {e}")
                    # If failed, pad with zeros
                    tactile_data.extend([0.0] * 128)  # Assume each finger has 128 values
            
            # Normalize tactile data
            tactile_data = normalize_tactile_data(tactile_data)
            result['tactile'] = torch.FloatTensor(tactile_data)
        
        # 3. Load proprioception data if needed
        if 'proprioception' in self.modalities:
            motor_joints_path = sample_path / "motor_joints.txt"
            try:
                with open(motor_joints_path, 'r') as f:
                    lines = f.readlines()
                    joints = []
                    for line in lines:
                        line = line.strip()
                        if line:
                            joints.append(float(line))
            except Exception as e:
                print(f"Cannot load motor joints file {motor_joints_path}: {e}")
                joints = [180.0, 180.0, 180.0]  # Default value
            
            # Normalize motor joints
            joints = normalize_motor_joints(joints)
            result['motor_joints'] = torch.FloatTensor(joints)
        
        return result

def load_multimodal_dataset(root_dir):
    """Load multi-modal dataset"""
    sample_paths = []
    labels = []
    label_names = []
    
    print("Loading multi-modal dataset...")
    
    root_path = Path(root_dir)
    label_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    for idx, label_dir in enumerate(label_dirs):
        label_name = label_dir.name
        label_names.append(label_name)
        
        sample_count = 0
        for sample_folder in label_dir.iterdir():
            if sample_folder.is_dir():
                # Check if necessary files exist
                required_files = [
                    sample_folder / "gelsight_raw_image.png",
                    sample_folder / "finger1_tactile_array.txt",
                    sample_folder / "finger2_tactile_array.txt",
                    sample_folder / "finger3_tactile_array.txt",
                    sample_folder / "motor_joints.txt"
                ]
                
                if all(f.exists() for f in required_files):
                    sample_paths.append(str(sample_folder))
                    labels.append(idx)
                    sample_count += 1
        
        print(f"  {label_name}: {sample_count} samples")
    
    print(f"Loaded {len(sample_paths)} multi-modal samples, {len(label_names)} classes")
    return sample_paths, labels, label_names

# ---------- Utility Functions ----------
def reorder_labels_for_display(label_names):
    """Reorder labels: Can1-Can3 first, then alphabetical"""
    can_labels = []
    other_labels = []
    
    for i, label in enumerate(label_names):
        if label.startswith('Can') and len(label) > 3:
            try:
                can_num = int(label[3:])
                can_labels.append((can_num, i, label))
            except ValueError:
                other_labels.append((0, i, label))
        else:
            other_labels.append((0, i, label))
    
    can_labels.sort()
    other_labels.sort(key=lambda x: x[2])
    
    ordered_indices = [idx for _, idx, _ in can_labels + other_labels]
    ordered_labels = [label for _, _, label in can_labels + other_labels]
    
    return ordered_indices, ordered_labels

def plot_confusion_matrix(y_true, y_pred, label_names, title, modalities):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    ordered_indices, ordered_labels = reorder_labels_for_display(label_names)
    cm = confusion_matrix(y_true, y_pred)
    cm_reordered = cm[np.ix_(ordered_indices, ordered_indices)]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_reordered, display_labels=ordered_labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    im = ax.images[0]
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = "confusion_matrices"
    os.makedirs(output_dir, exist_ok=True)

    # Save the confusion matrix as SVG
    if len(modalities) == 1:
        modality = modalities[0]
        if modality == 'tactile_array':
            filename = f"single_{modality}_confusion_matrix.svg"
        else:
            filename = f"single_{modality}_confusion_matrix.svg"
    else:
        modality_str = "_".join(modalities)
        filename = f"{modality_str}_fusion_confusion_matrix.svg"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Confusion matrix saved as: {filepath}")

    plt.show()

    # Print per-class accuracy
    class_acc = cm_reordered.diagonal() / cm_reordered.sum(axis=1)
    print("Per-class accuracy:")
    for label, acc in zip(ordered_labels, class_acc):
        print(f"  {label}: {acc:.4f}")

def evaluate_model(model, dataloader, device, modalities):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {}
            for modality in modalities:
                if modality == 'tactile_image' and 'image' in batch:
                    inputs['image'] = batch['image'].to(device)
                elif modality == 'tactile_array' and 'tactile' in batch:
                    inputs['tactile'] = batch['tactile'].to(device)
                elif modality == 'proprioception' and 'motor_joints' in batch:
                    inputs['motor_joints'] = batch['motor_joints'].to(device)
            
            labels = batch['label'].to(device)
            
            outputs, _ = model(**inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

# ---------- Training Function ----------
def train(args):
    modalities = args.modalities
    dataset_dir = args.dataset_dir
    base_image_path = args.base_image_path
    save_model = getattr(args, 'save_model', False)
    
    is_single_modality = len(modalities) == 1
    
    print("="*60)
    if is_single_modality:
        print(f"Single Modality Training")
        print(f"Selected Modality: {modalities[0].upper()}")
        if modalities[0] == 'tactile_image':
            print(f"Image Encoder: ResNet")
        elif modalities[0] == 'tactile_array':
            print(f"Tactile Encoder: MLP")
        elif modalities[0] == 'proprioception':
            print(f"Motor Encoder: 3-layer MLP")
    else:
        print(f"Multi-Modal Fusion with Cross Attention")
        print(f"Selected Modalities: {' + '.join(modalities)}")
        if 'tactile_image' in modalities:
            print(f"Image Encoder: ResNet")
    print("="*60)
    
    # Load base image if image modality is used
    base_image_tensor = None
    if 'tactile_image' in modalities:
        base_image_tensor = load_base_image(base_image_path)
    
    # Load dataset
    sample_paths, labels, label_names = load_multimodal_dataset(dataset_dir)
    
    # Split into training and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        sample_paths, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"\nData split:")
    print(f"Training set: {len(train_paths)} samples")
    print(f"Test set: {len(test_paths)} samples")
    
    # Create datasets
    train_dataset = MultiModalDataset(train_paths, train_labels, modalities, base_image_tensor)
    test_dataset = MultiModalDataset(test_paths, test_labels, modalities, base_image_tensor)
    
    # Check tactile data dimension if tactile modality is used
    tactile_dim = 384  # Default
    if 'tactile_array' in modalities:
        sample_data = train_dataset[0]
        tactile_dim = sample_data['tactile'].shape[0]
        print(f"Tactile array data dimension: {tactile_dim}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(label_names)
    model = MultiModalFusionClassifier(
        num_classes, modalities, tactile_dim
    ).to(device)
    
    # Only train trainable parameters (non-frozen parts)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    print(f"\nStarting training...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LR}, Epochs: {NUM_EPOCHS}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Training loop
    best_train_acc = 0.0
    if is_single_modality:
        modality = modalities[0]
        if modality == 'tactile_array':
            model_filename = f'best_single_{modality}_mlp_model.pth'
        else:
            model_filename = f'best_single_{modality}_resnet_model.pth'
    else:
        modality_str = "_".join(modalities)
        model_filename = f'best_{modality_str}_resnet_fusion_model.pth'
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs = {}
            for modality in modalities:
                if modality == 'tactile_image' and 'image' in batch:
                    inputs['image'] = batch['image'].to(device)
                elif modality == 'tactile_array' and 'tactile' in batch:
                    inputs['tactile'] = batch['tactile'].to(device)
                elif modality == 'proprioception' and 'motor_joints' in batch:
                    inputs['motor_joints'] = batch['motor_joints'].to(device)
            
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs, attention_weights = model(**inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # Calculate accuracy
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Save best model
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            if save_model:
                torch.save(model.state_dict(), model_filename)
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.4f}")
    
    print(f"\nTraining completed! Best training accuracy: {best_train_acc:.4f}")
    
    # Load best model for testing (only if model was saved)
    if save_model:
        model.load_state_dict(torch.load(model_filename))
        print(f"Loaded best model from: {model_filename}")
    else:
        print("Model saving disabled, using current model for testing")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_labels_true, test_preds = evaluate_model(model, test_loader, device, modalities)
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(test_labels_true, test_preds)
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification report:")
    print(classification_report(test_labels_true, test_preds, target_names=label_names))
    
    # Plot confusion matrix
    print("\nConfusion matrix:")
    plot_confusion_matrix(test_labels_true, test_preds, label_names, 
                         "Classification Results", modalities)
    
    if save_model:
        print(f"\nModel saved as: {model_filename}")
    else:
        print("\nModel saving was disabled")
    print(f"Final test accuracy: {test_accuracy:.4f}")
        
    return model, test_accuracy

def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Fusion Classifier')
    parser.add_argument('--modalities', nargs='+', 
                       choices=['tactile_image', 'tactile_array', 'proprioception'],
                       help='Select modalities to use (e.g., --modalities tactile_image or --modalities tactile_image tactile_array)')
    parser.add_argument('--save_model', action='store_true', default=False,
                       help='Save the best model during training (default: False)')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                       help='Path to dataset directory')
    parser.add_argument('--base_image_path', type=str, default=DEFAULT_BASE_IMAGE_PATH,
                       help='Path to base image for difference calculation')
    
    args = parser.parse_args()
    
    # Validate arguments
    model, accuracy = train(args)
    print(f"\nTraining completed with accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main() 