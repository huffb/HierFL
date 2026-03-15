import io
import tempfile

import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from matplotlib import pyplot as plt
import pandas as pd
import recog
from models import initialize_model
from options import args_parser
from hierfavg import HierFAVG, training_state
from frontend_args import override_args
from models.initialize_model import initialize_model
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from flask_bootstrap import Bootstrap
import subprocess
from models.mnist_cnn import mnist_lenet
import os
import threading
import json

# 需要安装pandas和openpyxl库，随便什么版本都可以

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于会话管理
Bootstrap(app)

device = torch.device('cpu')
model = mnist_lenet(input_channels=1, output_channels=10)
model.load_state_dict(torch.load("D:/HierFL-master/trained_model/trained_model.pth"))
model.eval()

# 定义图像预处理的转换操作
transform = transforms.Compose([
    # 将图像转换为灰度图，如果图像本身已经是灰度图，此步骤可省略
    transforms.Grayscale(num_output_channels=1),
    # 调整图像大小为28x28像素
    transforms.Resize((28, 28)),
    # 将图像转换为张量
    transforms.ToTensor(),
    # 归一化处理，MNIST数据集的均值和标准差分别为0.1307和0.3081
    transforms.Normalize((0.1307,), (0.3081,))
])

# 模拟用户数据库
users = {
    'admin': 'password'
}


@app.route('/login', methods=['GET', 'POST'])
def login():
    # if request.method == 'POST':
    #     username = request.form.get('username')
    #     password = request.form.get('password')
    #     if username in users and users[username] == password:
    #         session['username'] = username
    #         return redirect(url_for('index'))
    #     else:
    #         return 'Invalid username or password'
    # return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
def upload_file():
    # if 'username' not in session:
    #     return redirect(url_for('login'))
    file = request.files['image']
    if file:
        try:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            return jsonify({'status': 'success', 'file_path': 'uploads/' + file.filename})
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'status': 'error'})
    return jsonify({'status': 'error'})


@app.route('/')
def index():
    # if 'username' not in session:
    #     return redirect(url_for('login'))
    return render_template("login.html")


@app.route('/train')
def train():
    # if 'username' not in session:
    #     return redirect(url_for('login'))
    return render_template("train.html")


@app.route('/start_training', methods=['POST'])
def start_training():
    frontend_data = request.get_json()
    print("Received data:", frontend_data)
    args = override_args(frontend_data)
    print(args)
    # 启动训练线程，避免阻塞响应
    training_state["progress"] = 0
    training_thread = threading.Thread(target=HierFAVG, args=(args,))
    training_thread.start()
    # 训练完成后返回 JSON 响应
    return jsonify({"status": "started"})


@app.route('/get_progress')
def get_progress():
    return jsonify({"progress": training_state["progress"]})


# 点击按钮保存模型，自动下载。
@app.route('/download_model', methods=['GET'])
def download_model():
    model_path = "D:/work/hfl/expirentment/trained_model.pth"  # 你实际的模型路径
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return {"error": "模型文件不存在"}, 404


# 注意地址变化，用于在前端生成训练结果图
@app.route('/generate_image')
def generate_temp_image():
    # Excel 文件路径
    excel_path = 'D:/work/hfl/expirentment/DP/fmnist.xlsx'  # 请确认文件在后端根目录下
    # 读取 Excel 文件
    df = pd.read_excel(excel_path)
    # 提取准确率和验证损失数据（Acc 和 Loss）
    accuracy_data = df['Acc'].tolist()
    validation_loss_data = df['Loss'].tolist()
    plt.rcParams['font.sans-serif'] = ['SimSun']
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    # 转换为百分制
    accuracy_percentage = [acc * 100 for acc in accuracy_data]
    # 设置横坐标
    x = np.arange(1, len(accuracy_percentage) + 1)

    # 创建一个包含两个子图的画布
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制准确率图
    # 设置横坐标间隔为5
    axes[0].tick_params(axis='x', labelsize=14);
    axes[0].tick_params(axis='y', labelsize=14);
    axes[0].set_xticks(np.arange(1, len(accuracy_percentage) + 1, 5))
    # 设置纵坐标间隔为10
    axes[0].set_yticks(np.arange(0, 101, 10))
    axes[0].plot(x, accuracy_percentage, marker='o', linestyle='-')
    axes[0].set_xlabel('全局聚合轮次', fontsize=20)
    axes[0].set_ylabel('准确率(%)', fontsize=20)
    axes[0].set_title('模型准确率曲线', fontsize=20)
    axes[0].grid(True)

    # 绘制验证损失图
    axes[1].tick_params(axis='x', labelsize=14);
    axes[1].tick_params(axis='y', labelsize=14);
    axes[1].set_xticks(np.arange(1, len(validation_loss_data) + 1, 5))
    axes[1].set_yticks(np.arange(0, 2.5,0.2 ))
    axes[1].plot(x, validation_loss_data, marker='o', linestyle='-')
    axes[1].set_xlabel('全局聚合轮次', fontsize=20)
    axes[1].set_ylabel('验证损失', fontsize=20)
    axes[1].set_title('模型验证损失曲线', fontsize=20)
    axes[1].grid(True)

    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp()
    # 创建图片并保存到临时文件夹
    img_path = os.path.join(temp_dir, 'temp_image.jpg')
    plt.savefig(img_path)

    # 返回生成的图片
    return send_file(img_path, mimetype='image/jpeg')


@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    # if 'username' not in session:
    #     return redirect(url_for('login'))
    if request.method == 'GET':
        return render_template("recognize.html")
    elif request.method == 'POST':
        data = request.get_json()  # 获取 JSON 数据
        if data and 'filepath' in data:
            file_path = data['filepath']
            try:
                image = Image.open(file_path)
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output.data, 1)
                    result = predicted.item()
                    print(f"预测结果类别索引: {result}")
                return jsonify({"status": "success", 'result': result})
            except Exception as e:
                print(f"Error: {e}")
                return jsonify({'result': '识别失败'})
        else:
            return jsonify({'result': '未提供文件路径'})


# 图片路径一定要是英文的
@app.route('/recognize_img')
def recognize_img():
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, 'recognized_image.jpg')
    model_path = "D:/HierFL-master/trained_model/trained_model.pth"
    recognizer = recog.DigitRecognizer(model_path)
    file_path = request.args.get('filepath')
    file_path = file_path.replace('\\', '/')
    file_path = os.path.join(app.root_path, *file_path.split('/'))
    results = recognizer.process_image(file_path, temp_output_path)

    if results:
        result_save_path = "D:/HierFL-master/templates/recognized_result.json"   # 分类结果保存地址
        with open(result_save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return send_file(temp_output_path, mimetype='image/jpeg')
    else:
        return jsonify({'status': 'error', 'message': '识别失败'})


# 下载分类结果
@app.route('/download_result')
def download_result():
    path = "D:/HierFL-master/templates/recognized_result.json"
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)