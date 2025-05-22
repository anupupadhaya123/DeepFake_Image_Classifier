from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from predict import predict_mobilenet, predict_vit
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['image']
        model_type = request.form['model']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            image = Image.open(filepath).convert("RGB")

            if model_type == 'mobilenet':
                result, confidence = predict_mobilenet(image)
            else:
                result, confidence = predict_vit(image)

            os.remove(filepath)

    return render_template('index.html', result=result, confidence=confidence, heatmap_path="/static/heatmap.jpg")


if __name__ == '__main__':
    app.run(debug=True)