import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_bootstrap import Bootstrap
from matplotlib import pyplot as plt

import recog
from frontend_args import override_args
from hierfavg import HierFAVG, training_state
from models.mnist_cnn import mnist_lenet


app = Flask(__name__)
app.secret_key = 'your_secret_key'
Bootstrap(app)

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
RESULTS_DIR = ARTIFACTS_DIR / "results"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
DEFAULT_MODEL_PATH = MODELS_DIR / "trained_model.pth"
RECOGNIZED_RESULT_PATH = RESULTS_DIR / "recognized_result.json"

device = torch.device('cpu')

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

users = {
    'admin': 'password'
}


def load_inference_model():
    if not DEFAULT_MODEL_PATH.exists():
        return None

    inference_model = mnist_lenet(input_channels=1, output_channels=10)
    inference_model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location=device))
    inference_model.eval()
    return inference_model


def get_latest_metrics_path():
    if not METRICS_DIR.exists():
        return None

    metric_files = sorted(METRICS_DIR.glob("*_training_metrics.xlsx"), key=lambda path: path.stat().st_mtime)
    if not metric_files:
        return None
    return metric_files[-1]


@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('index.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    if file:
        try:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = UPLOADS_DIR / file.filename
            file.save(file_path)
            return jsonify({'status': 'success', 'file_path': str(Path('uploads') / file.filename)})
        except Exception as exc:
            print(f"Error: {exc}")
            return jsonify({'status': 'error'})
    return jsonify({'status': 'error'})


@app.route('/')
def index():
    return render_template("login.html")


@app.route('/train')
def train():
    return render_template("Train.html")


@app.route('/start_training', methods=['POST'])
def start_training():
    frontend_data = request.get_json()
    print("Received data:", frontend_data)
    args = override_args(frontend_data)
    print(args)
    training_state["progress"] = 0
    training_thread = threading.Thread(target=HierFAVG, args=(args,))
    training_thread.start()
    return jsonify({"status": "started"})


@app.route('/get_progress')
def get_progress():
    return jsonify({"progress": training_state["progress"]})


@app.route('/download_model', methods=['GET'])
def download_model():
    if DEFAULT_MODEL_PATH.exists():
        return send_file(DEFAULT_MODEL_PATH, as_attachment=True)
    return {"error": "模型文件不存在"}, 404


@app.route('/generate_image')
def generate_temp_image():
    metrics_path = get_latest_metrics_path()
    if metrics_path is None:
        return {"error": "训练结果文件不存在"}, 404

    df = pd.read_excel(metrics_path)
    accuracy_data = df['Acc'].tolist()
    validation_loss_data = df['Loss'].tolist()
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    accuracy_percentage = [acc * 100 for acc in accuracy_data]
    x = np.arange(1, len(accuracy_percentage) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].set_xticks(np.arange(1, len(accuracy_percentage) + 1, 5))
    axes[0].set_yticks(np.arange(0, 101, 10))
    axes[0].plot(x, accuracy_percentage, marker='o', linestyle='-')
    axes[0].set_xlabel('全局聚合轮次', fontsize=20)
    axes[0].set_ylabel('准确率(%)', fontsize=20)
    axes[0].set_title('模型准确率曲线', fontsize=20)
    axes[0].grid(True)

    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].set_xticks(np.arange(1, len(validation_loss_data) + 1, 5))
    axes[1].set_yticks(np.arange(0, 2.5, 0.2))
    axes[1].plot(x, validation_loss_data, marker='o', linestyle='-')
    axes[1].set_xlabel('全局聚合轮次', fontsize=20)
    axes[1].set_ylabel('验证损失', fontsize=20)
    axes[1].set_title('模型验证损失曲线', fontsize=20)
    axes[1].grid(True)

    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, 'temp_image.jpg')
    plt.savefig(img_path)
    plt.close(fig)
    return send_file(img_path, mimetype='image/jpeg')


@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'GET':
        return render_template("recognize.html")

    data = request.get_json()
    if data and 'filepath' in data:
        file_path = data['filepath']
        try:
            inference_model = load_inference_model()
            if inference_model is None:
                return jsonify({'result': '模型文件不存在，请先训练或放置模型'})

            image = Image.open(PROJECT_ROOT / file_path)
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = inference_model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                result = predicted.item()
                print(f"预测结果类别索引: {result}")
            return jsonify({"status": "success", 'result': result})
        except Exception as exc:
            print(f"Error: {exc}")
            return jsonify({'result': '识别失败'})

    return jsonify({'result': '未提供文件路径'})


@app.route('/recognize_img')
def recognize_img():
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, 'recognized_image.jpg')
    if not DEFAULT_MODEL_PATH.exists():
        return jsonify({'status': 'error', 'message': '模型文件不存在，请先训练或放置模型'})

    recognizer = recog.DigitRecognizer(DEFAULT_MODEL_PATH)
    file_path = request.args.get('filepath')
    file_path = file_path.replace('\\', '/')
    file_path = os.path.join(app.root_path, *file_path.split('/'))
    results = recognizer.process_image(file_path, temp_output_path)

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RECOGNIZED_RESULT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return send_file(temp_output_path, mimetype='image/jpeg')

    return jsonify({'status': 'error', 'message': '识别失败'})


@app.route('/download_result')
def download_result():
    if not RECOGNIZED_RESULT_PATH.exists():
        return {"error": "识别结果不存在"}, 404
    return send_file(RECOGNIZED_RESULT_PATH, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
