import numpy as np
import torch
from tensorflow.keras.models import load_model
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import cv2

# Load models once
mobilenet_model = load_model("models/deepfake_detection_mobileNet_model.h5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
vit_model.classifier = torch.nn.Linear(vit_model.config.hidden_size, 2)
vit_model.load_state_dict(torch.load("models/vit_real_fake_model.pth", map_location=device))
vit_model = vit_model.to(device)
vit_model.eval()

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess_mobilenet(img):
    img = img.resize((96, 96))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_vit(img):
    inputs = processor(images=img, return_tensors="pt")
    return inputs['pixel_values'].to(device)

def predict_mobilenet(img):
    arr = preprocess_mobilenet(img)
    pred = mobilenet_model.predict(arr)
    class_idx = np.argmax(pred, axis=1)[0]
    return ("Fake" if class_idx == 0 else "Real"), float(np.max(pred))

def predict_vit(img):
    inputs = preprocess_vit(img)
    with torch.no_grad():
        outputs = vit_model(pixel_values=inputs)
        class_idx = torch.argmax(outputs.logits, dim=1).item()
        prob = torch.nn.functional.softmax(outputs.logits, dim=1)[0, class_idx].item()

    generate_attention_heatmap(img, vit_model, processor)
    return ("Fake" if class_idx == 0 else "Real"), prob


# ViT Attention Heatmap Generator
def generate_attention_heatmap(image: Image, model, processor, save_path="static/heatmap.jpg"):
    model.eval()
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Extract raw attention from last layer
    with torch.no_grad():
        outputs = model.vit(pixel_values=pixel_values, output_attentions=True)
        attn = outputs.attentions[-1]  # last attention layer (batch, heads, tokens, tokens)

    # Get attention for class token to all patches
    cls_attn = attn[0, :, 0, 1:]  # (heads, 196)
    cls_attn = cls_attn.mean(dim=0)  # (196,)

    # Reshape to 14x14 and resize to image size
    attn_map = cls_attn.reshape(14, 14).cpu().numpy()
    attn_map = cv2.resize(attn_map, image.size)

    # Normalize and convert to heatmap
    norm_attn = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    heatmap = np.uint8(255 * norm_attn)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Merge heatmap with original image
    orig = np.array(image)
    if orig.shape[-1] == 4:
        orig = cv2.cvtColor(orig, cv2.COLOR_RGBA2RGB)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)
