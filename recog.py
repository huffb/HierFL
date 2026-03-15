import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from models.mnist_cnn import mnist_lenet


# 修正的模型定义
class DigitRecognizer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _load_model(self, model_path):
        model = mnist_lenet(input_channels=1, output_channels=10)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

    def _preprocess_image(self, image_path):
        # 使用OpenCV读取并预处理
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return img, sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    def _predict_digit(self, roi):
        image = Image.fromarray(roi).convert('L')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        return torch.argmax(output).item()

    def process_image(self, input_path, output_path):
        try:
            # 预处理和检测
            orig_img, contours = self._preprocess_image(input_path)
            pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # 加载字体，这里假设系统中有 Arial 字体，可根据实际情况修改
            font = ImageFont.truetype("arial.ttf", 30)

            results = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < 100:  # 过滤小区域
                    continue

                # 裁剪数字区域
                roi = orig_img[y:y + h, x:x + w]
                digit = self._predict_digit(roi)

                # 标注图像，增加框的宽度
                draw.rectangle([x, y, x + w, y + h], outline='red', width=5)
                draw.text((x + 5, y - 35), str(digit), fill='red', font=font)
                results.append({
                    'digit': digit,
                    'position': (x, y, w, h)
                })

            # 保存结果
            pil_img.save(output_path)
            print(f"结果已保存至: {output_path}")
            return results

        except Exception as e:
            print(f"处理失败: {str(e)}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MNIST数字识别')
    parser.add_argument('-i', '--input', default="D:/HierFL-master/uploads/1324.jpg", help='输入图像路径')
    parser.add_argument('-o', '--output', default="D:/work/hfl/output.png", help='输出图像路径')
    parser.add_argument('-m', '--model', default="D:/HierFL-master/trained_model/trained_model.pth",
                        help='模型文件路径')

    args = parser.parse_args()

    # 检查输入图像路径是否存在
    if not args.input:
        raise ValueError("输入图像路径为空")

    # 检查模型文件路径是否存在
    try:
        with open(args.model, 'r'):
            pass
    except FileNotFoundError:
        raise FileNotFoundError(f"模型文件未找到: {args.model}")

    recognizer = DigitRecognizer(args.model)
    results = recognizer.process_image(args.input, args.output)

    if results:
        print("识别结果：")
        for idx, res in enumerate(results, 1):
            print(f"数字{idx}: 值={res['digit']}, 位置(x,y,w,h)=({res['position']})")
