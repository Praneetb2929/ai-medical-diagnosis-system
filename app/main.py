import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI(title="AI Medical Diagnosis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["NORMAL", "PNEUMONIA"]

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# ---------- Baseline CNN architecture (must match models/baseline_cnn.py) ----------
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def load_baseline():
    model = BaselineCNN()
    model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

def load_densenet():
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load("densenet_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

baseline_model = load_baseline()
densenet_model = load_densenet()

baseline_cam = GradCAM(model=baseline_model, target_layers=[baseline_model.conv_layers[6]])
densenet_cam = GradCAM(model=densenet_model, target_layers=[densenet_model.features.denseblock4.denselayer16.conv2])

UNCERTAIN_THRESHOLD = 0.65  # if top confidence is below this, flag as uncertain

# ---------- Hardcoded evaluation results (from your Colab training runs) ----------
METRICS = {
    "baseline": {
        "accuracy": 0.78,
        "sensitivity": 0.98,
        "specificity": 0.18,
        "auc_roc": None,
        "confusion_matrix": [[43, 191], [1, 389]],
        "note": "High run-to-run variance observed (NORMAL recall ranged 0.18-0.44 across runs) due to unweighted loss on an imbalanced dataset."
    },
    "densenet121": {
        "accuracy": 0.87,
        "sensitivity": 0.94,
        "specificity": 0.76,
        "auc_roc": 0.9405,
        "confusion_matrix": [[178, 56], [24, 366]],
        "note": "Class-weighted loss corrected the baseline's imbalance bias while preserving strong pneumonia recall."
    }
}

def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return np.clip(img, 0, 1)

def image_to_base64(np_img):
    img = Image.fromarray((np_img * 255).astype(np.uint8)) if np_img.max() <= 1 else Image.fromarray(np_img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_model(model, cam, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    rgb_img = unnormalize(input_tensor[0])
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    confidence = {"NORMAL": float(probs[0]), "PNEUMONIA": float(probs[1])}
    top_conf = max(confidence.values())

    return {
        "prediction": classes[pred_idx],
        "confidence": confidence,
        "uncertain": top_conf < UNCERTAIN_THRESHOLD,
        "gradcam_image_base64": image_to_base64(cam_image)
    }

@app.get("/")
def root():
    return {"status": "AI Medical Diagnosis System API is running"}

@app.get("/metrics")
def get_metrics():
    return JSONResponse(METRICS)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    densenet_result = run_model(densenet_model, densenet_cam, input_tensor)
    baseline_result = run_model(baseline_model, baseline_cam, input_tensor)

    return JSONResponse({
        "densenet121": densenet_result,
        "baseline": baseline_result
    })