"""
CosPlace Visual Place Recognition utilities for Netryx.
Handles model loading, descriptor extraction, and index operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

_cosplace_model = None
_cosplace_transform = None

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def get_cosplace_model():
    """Load CosPlace model (ResNet50, 512-dim). Cached after first call."""
    global _cosplace_model, _cosplace_transform

    if _cosplace_model is not None:
        return _cosplace_model

    print("[CosPlace] Loading model (ResNet50, 512-dim)...")
    _cosplace_model = torch.hub.load(
        'gmberton/cosplace',
        'get_trained_model',
        backbone='ResNet50',
        fc_output_dim=512,
        trust_repo=True
    ).eval().to(device)

    _cosplace_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("[CosPlace] Model loaded successfully.")
    return _cosplace_model


def get_cosplace_transform():
    global _cosplace_transform
    if _cosplace_transform is None:
        get_cosplace_model()
    return _cosplace_transform


def extract_cosplace_descriptor(image):
    """
    Extract a 512-dim L2-normalized CosPlace descriptor from a PIL image.
    Returns: numpy array of shape (512,)
    """
    model = get_cosplace_model()
    transform = get_cosplace_transform()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        descriptor = model(tensor)
        descriptor = F.normalize(descriptor, p=2, dim=1)

    return descriptor.cpu().numpy().flatten()


def batch_extract_cosplace(images, batch_size=8):
    """
    Extract CosPlace descriptors for a list of PIL images.
    Returns: numpy array of shape (N, 512)
    """
    model = get_cosplace_model()
    transform = get_cosplace_transform()

    all_descs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = []
        for img in batch:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            tensors.append(transform(img))

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            descs = model(batch_tensor)
            descs = F.normalize(descs, p=2, dim=1)
        all_descs.append(descs.cpu().numpy())

    return np.vstack(all_descs) if all_descs else np.zeros((0, 512))


def load_cosplace_index(path):
    """Load a CosPlace index file. Returns (descriptors, paths)."""
    data = np.load(path, allow_pickle=True)
    return data['descriptors'], list(data['paths'])


def save_cosplace_index(descriptors, paths, path):
    """Save CosPlace index."""
    np.savez(path, descriptors=descriptors, paths=np.array(paths, dtype=object))


def cosplace_similarity(q_feat, index_feats):
    """
    Compute cosine similarity between a query vector and a matrix of descriptors.
    q_feat: (512,)
    index_feats: (N, 512)
    Returns: (N,)
    """
    if len(q_feat.shape) == 1:
        q_feat = q_feat.reshape(1, -1)
    
    # Assume both are already L2 normalized
    sim = np.dot(index_feats, q_feat.T).flatten()
    return sim
