from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from models.mnist_cnn import mnist_lenet


class DigitRecognizer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def _load_model(self, model_path):
        model = mnist_lenet(input_channels=1, output_channels=10)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _preprocess_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return img, sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

    def _predict_digit(self, roi):
        image = Image.fromarray(roi).convert('L')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        return torch.argmax(output).item()

    def process_image(self, input_path, output_path):
        try:
            orig_img, contours = self._preprocess_image(input_path)
            pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except OSError:
                font = ImageFont.load_default()

            results = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h < 100:
                    continue

                roi = orig_img[y:y + h, x:x + w]
                digit = self._predict_digit(roi)
                draw.rectangle([x, y, x + w, y + h], outline='red', width=5)
                draw.text((x + 5, y - 35), str(digit), fill='red', font=font)
                results.append(
                    {
                        'digit': digit,
                        'position': (x, y, w, h),
                    }
                )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(output_path)
            print(f"结果已保存至: {output_path}")
            return results
        except Exception as exc:
            print(f"处理失败: {exc}")
            return None


if __name__ == "__main__":
    import argparse

    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description='MNIST数字识别')
    parser.add_argument(
        '-i',
        '--input',
        default=str(project_root / "uploads" / "1324.jpg"),
        help='输入图像路径',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=str(project_root / "artifacts" / "results" / "recognized_image.png"),
        help='输出图像路径',
    )
    parser.add_argument(
        '-m',
        '--model',
        default=str(project_root / "artifacts" / "models" / "trained_model.pth"),
        help='模型文件路径',
    )

    args = parser.parse_args()
    recognizer = DigitRecognizer(args.model)
    results = recognizer.process_image(args.input, args.output)

    if results:
        print("识别结果:")
        for idx, res in enumerate(results, 1):
            print(f"数字{idx}: 值={res['digit']}, 位置(x,y,w,h)=({res['position']})")
