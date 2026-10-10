import io                                      
from pathlib import Path                       
import torch                                   
import torchvision.transforms as transforms    
from PIL import Image                           

from app.cnn_model import CNN                  

# Locate the weights file relative to THIS file, not the current working directory
WEIGHTS_PATH = Path(__file__).parent / "cnn_cifar10.pth"                

# CIFAR10 class names, in the same order as the training labels (0-9)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Build the model once, load the trained weights, switch to evaluation mode
# Runs once when the server starts, not on every request
model = CNN()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
model.eval()

# Preprocessing must match training 
# resize to 64x64, convert to a tensor in [0, 1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def predict(image_bytes: bytes) -> dict:
    """Takes raw image bytes and returns the predicted class and confidence."""
    # Open the image and force 3 color channels 
    # handles PNGs with transparency, grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess and add a batch dimension: [3, 64, 64] -> [1, 3, 64, 64]
    x = transform(image).unsqueeze(0)

    # Forward pass with no gradient tracking (inference only)
    with torch.no_grad():
        logits = model(x)                           # raw scores, shape [1, 10]
        probs = torch.softmax(logits, dim=1)[0]     # convert scores to probabilities summing to 1

    # Pick the class with the highest probability
    top_idx = int(torch.argmax(probs))

    return {
        "predicted_class": CLASSES[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }