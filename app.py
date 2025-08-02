from flask import Flask, request, render_template
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch
import torch.nn as nn
from PIL import Image
import os

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model.heads.head = nn.Linear(model.heads.head.in_features, 4)  # change 4 if different
model.load_state_dict(torch.load("D:/mini project/VIT/Model Frontend/best_lung_disease_vit.pth", map_location=device))
model.eval()
model.to(device)

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels (adjust if different)
classes = ['Bacterial Pneumonia', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    image = Image.open(file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        result = classes[predicted.item()]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
