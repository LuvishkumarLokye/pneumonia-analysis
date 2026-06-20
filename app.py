from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import os

app = Flask(__name__)

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN()
MODEL_PATH = "best_pneumonia_model.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"[!] Warning: {MODEL_PATH} missing. Inference metrics will output random distributions.")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image payload received."}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
        
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Fixed: Force color decoding directly to prevent grayscale dimension alignment breaks
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image format."}), 400

        # Sync colorspace parameters to match tensor expectation frameworks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(rgb_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        class_names = ["NORMAL", "PNEUMONIA"]
        class_idx = predicted.item()

        return jsonify({
            "prediction": class_names[class_idx],
            "confidence": round(probabilities[0][class_idx].item() * 100, 2),
            "prob_normal": round(probabilities[0][0].item() * 100, 2),
            "prob_pneumonia": round(probabilities[0][1].item() * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": f"Inference engine fault: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)