import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from tensorflow import keras

# Define transformations for MobileNetV2
mobile_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_mobilenet_features(img_path: str, model_path: str, device: torch.device) -> np.ndarray:
    """Extracts 1×1280 feature vectors from an image using a trained MobileNetV2 model."""
    mobilenet = models.mobilenet_v2()
    num_features = mobilenet.classifier[1].in_features
    mobilenet.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    mobilenet.load_state_dict(torch.load(model_path, map_location=device))
    mobilenet.eval().to(device)

    img = Image.open(img_path).convert('RGB')
    img_tensor = mobile_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = mobilenet.features(img_tensor)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return pooled.flatten(1).cpu().numpy()  # shape (1, 1280)


def divide_and_average(arr: np.ndarray, n: int) -> list:
    avg_values = []
    part_size = len(arr) // n
    for i in range(n):
        start = i * part_size
        end = start + part_size if i < n - 1 else len(arr)
        avg_values.append(float(arr[start:end].mean()))
    return avg_values


def sbtcFeatures(image_path: str, n: int = 10) -> list:
    """Compute sorted block BTC features: returns 3*n averages for R, G, B channels."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image at {image_path}")

    b, g, r = cv2.split(img)
    features = []
    for channel in (r, g, b):
        arr = channel.ravel()
        arr.sort()
        features.extend(divide_and_average(arr, n))
    return features


def predict_malignant_benign(
    image_path: str,
    mobilenet_model_path: str,
    mlp_model_path: str,
    n_blocks: int = 10
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. MobileNetV2 features
    mobile_feats = extract_mobilenet_features(image_path, mobilenet_model_path, device)
    if mobile_feats.size == 0:
        return {"error": "MobileNetV2 feature extraction failed."}

    # 2. TSBTC features
    tsbtc_list = sbtcFeatures(image_path, n=n_blocks)
    tsbtc_feats = np.array(tsbtc_list, dtype=float).reshape(1, -1)
    if tsbtc_feats.size == 0:
        return {"error": "TSBTC feature extraction failed."}

    # 3. Combine features
    combined = np.concatenate((mobile_feats, tsbtc_feats), axis=1)

    # 4. Load MLP
    try:
        mlp = keras.models.load_model(mlp_model_path)
    except Exception as e:
        return {"error": f"Failed to load MLP: {e}"}

    # 5. Prediction
    try:
        probs = mlp.predict(combined)
        # Handle single-output (sigmoid) or two-output (softmax)
        if probs.shape[1] == 1:
            prob_malignant = float(probs[0, 0])
            prob_benign = 1.0 - prob_malignant
        else:
            prob_benign = float(probs[0, 0])
            prob_malignant = float(probs[0, 1])

        predicted = 'malignant' if prob_malignant > prob_benign else 'benign'

        return {
            "predicted_class":      predicted,
            "probability_benign":   f"{prob_benign:.4f}",
            "probability_malignant":f"{prob_malignant:.4f}"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}


if __name__ == '__main__':
    # Example usage
    image_path = "/content/drive/MyDrive/fyproject/breakhis_aug/Breakhis_400x/Malignant/SOB_M_DC-14-11520-400-019.png"
    mobilenet_path = "/content/drive/MyDrive/fyproject/mobilenet_unfreeze_6_adamw.pth"
    mlp_path       = "/content/drive/MyDrive/fyproject/mlp_breast_cancer_model.h5"

    result = predict_malignant_benign(
        image_path,
        mobilenet_path,
        mlp_path,
        n_blocks=10
    )

    if 'error' in result:
        print("Error:", result['error'])
    else:
        print(f"Prediction: {result['predicted_class']}")
        print(f"  Benign: {result['probability_benign']}")
        print(f"  Malignant: {result['probability_malignant']}")
