# Deepfake Detection System – User Guide

## Purpose

This User Guide serves two key goals:
1. To describe **how to use** the Deepfake Detection AI System from a user’s perspective.
2. To provide detailed instructions so a **developer or lecturer** can fully reproduce the system, from setup to prediction and result verification.

---

## System Description

The Deepfake Detector is a browser-based AI tool built using **Flask**, enabling users to:
- Upload a face image
- Choose between two AI models:
  - ✅ MobileNetV2 (Keras)
  - ✅ Vision Transformer (HuggingFace)
- Get a prediction of whether the face is **real or fake**
- Visualize **attention heatmaps** (XAI) using ViT
- Download detection results and view AI focus

---

## Local Setup Instructions

### 1. Dependencies
Install the required libraries:
```bash
pip install flask torch torchvision transformers pillow opencv-python
```

### 2. Project Structure
```
project/
├── app.py                    # Flask application logic
├── predict.py                # Inference and heatmap generation
├── templates/
│   └── index.html            # Web interface (HTML + Jinja2)
├── static/
│   ├── style.css             # External styling
│   └── heatmap.jpg           # AI-generated attention map
├── models/
│   ├── vit_real_fake_model.pth    # Pretrained ViT
│   └── deepfake_mobilenet.h5      # Keras MobileNetV2 model
```

### 3. Models Used

| Model          | Framework  | Format              | Dataset Used                     |
|----------------|------------|---------------------|----------------------------------|
| MobileNetV2    | Keras/TensorFlow | `.h5`        | FaceForensics (Real vs Fake)     |
| Vision Transformer (ViT) | PyTorch (HuggingFace) | `.pth` | Pretrained on CelebDF/Faces HQ  |

---

### ▶ 4. Running the App

In the terminal:
```bash
python app.py
```
Then open your browser at:
```
http://localhost:5000
```

---

## How to Use the Web App

1. **Upload an image** (only face images recommended)
2. **Choose a model** (Vision Transformer or MobileNetV2)
3. Click **Detect**
4. You’ll see:
   - A color-coded prediction card (Real or Fake)
   - Confidence score
   - Downloadable `.txt` result
   - ViT Heatmap (only if Vision Transformer is selected)
5. Go to back page for next detection

---

## Dataset Handling

- You must **separately prepare datasets** (e.g., FaceForensics++).
- MobileNet model was trained by splitting:
  - 80% training
  - 10% validation
  - 10% test
- Vision Transformer was **pretrained** and fine-tuned only on face classification.

---

## Screenshots to Reproduce

You must provide these:
1. Model file locations (`models/`)
2. Flask terminal output (`python app.py`)
3. Browser with uploaded image + prediction
4. Downloaded result file
5. ViT heatmap shown on screen

---

## How to Reproduce Predictions

To reproduce your result:
1. Place the same `.h5` and `.pth` files in `models/`
2. Upload the same image via the browser
3. Choose the correct model
4. Run inference and verify result
5. Ensure `static/heatmap.jpg` regenerates upon each ViT request

---

## Version Details

| Tool              | Version         |
|-------------------|-----------------|
| Python            | 3.11.x          |
| Flask             | 2.x             |
| Transformers      | 4.x             |
| Torch             | 2.x             |
| OpenCV            | 4.7+            |
| Pillow            | 10+             |

---

## Important Notes

- Make sure to clear browser cache if heatmap doesn't update visually.
- ViT model output may slightly vary based on attention sampling.
- The download result feature uses client-side JS, so no server file is saved.
- If the results cannot be regenerated with your `.h5`/`.pth`, marks may be lost.