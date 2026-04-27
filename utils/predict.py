import torch
from torchvision import models, transforms
from PIL import Image

# Vortrainiertes Modell laden
model = models.resnet18(pretrained=True)
model.eval()

# ImageNet Labels (vereinfachte Demo)
labels = ["motorcycle", "car", "bicycle", "truck"]

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

    _, predicted = outputs.max(1)

    # FAKE Mapping (nur Demo!)
    label = labels[predicted.item() % len(labels)]
    confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

    return label, confidence
