import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["NORMAL", "PNEUMONIA"]

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def load_model():
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load("densenet_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
target_layers = [model.features.denseblock4.denselayer16.conv2]
cam = GradCAM(model=model, target_layers=target_layers)

def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return np.clip(img, 0, 1)

def image_to_base64(np_img):
    img = Image.fromarray((np_img * 255).astype(np.uint8)) if np_img.max() <= 1 else Image.fromarray(np_img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/")
def root():
    return {"status": "AI Medical Diagnosis System API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    rgb_img = unnormalize(input_tensor[0])
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return JSONResponse({
        "prediction": classes[pred_idx],
        "confidence": {
            "NORMAL": float(probs[0]),
            "PNEUMONIA": float(probs[1])
        },
        "gradcam_image_base64": image_to_base64(cam_image)
    })