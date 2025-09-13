import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
import io
import os

# ------------------------
# Device & Classes
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['roads', 'streetlights', 'garbage', 'not_civic']

# ------------------------
# Model Definition
# ------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*56*56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# ------------------------
# Load Model
# ------------------------
model = SimpleCNN(num_classes=len(class_names))
# Place your .pth file in the same folder and load it
model.load_state_dict(torch.load('civic_issue_classifier.pth', map_location=device))
model.to(device)
model.eval()

# ------------------------
# Image Transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ------------------------
# Flask App
# ------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Civic Issue Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        class_idx = pred.item()
        class_name = class_names[class_idx]
    
    return jsonify({"prediction": class_name})

# ------------------------
# Run App
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
