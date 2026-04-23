import torch
import torchvision.transforms as transforms
from torchvision import models
from huggingface_hub import hf_hub_download
from PIL import Image
import io, json
from pathlib import Path

# Oxford Pets 37 breeds (alphabetical order matching training)
BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier"
]

# ImageNet normalization (same as training)
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model = None  # singleton


def load_model() -> torch.nn.Module:
    """Download checkpoint from HuggingFace Hub and load ResNet50."""
    global _model, BREEDS
    
    if _model is not None:
        return _model

    checkpoint_path = hf_hub_download(
        repo_id="flaviodell/oxford-pets-resnet50",
        filename="best_model.pth",
        cache_dir="/tmp/hf_cache"
    )

    # Rebuild the same architecture used in project "llm-cv-finetuning-pipeline"
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(model.fc.in_features, len(BREEDS)),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict and class list from checkpoint
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        # Override BREEDS with the exact class order used during training
        BREEDS = checkpoint["classes"]
    else:
        state_dict = checkpoint  # raw state_dict fallback

    model.load_state_dict(state_dict)

    model.eval()
    _model = model
    return _model


def predict(image_bytes: bytes) -> dict:
    """Run inference on raw image bytes. Returns breed + confidence."""
    model = load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top5_idx = probs.topk(5).indices.tolist()
    top5 = [{"breed": BREEDS[i], "confidence": round(probs[i].item(), 4)}
            for i in top5_idx]

    return {
        "breed": top5[0]["breed"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
    }
