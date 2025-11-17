import argparse, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------
# Model: simple CNN
# -----------------------
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Conv + ReLU + Pool layers
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# Chuẩn hoá chuẩn MNIST
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_set = datasets.MNIST(root="data", train=True, download=True, transform=train_tf)
    test_set  = datasets.MNIST(root="data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    model = MNISTCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = Path(args.save_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc  += (out.argmax(1) == y).float().sum().item()
            n += bs

        train_loss = total_loss / n
        train_acc = total_acc / n

        # evaluate
        model.eval()
        te_loss, te_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                bs = x.size(0)
                te_loss += loss.item() * bs
                te_acc  += (out.argmax(1) == y).float().sum().item()
                m += bs
        te_loss /= m
        te_acc  /= m

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} acc={train_acc:.4f} | test_loss={te_loss:.4f} acc={te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict()}, save_path)
            print(f"  ✓ Saved best model to {save_path} (acc={best_acc:.4f})")

# ========= Single-digit preprocess (OpenCV) =========
def preprocess_mnist_opencv(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2]

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bg = cv2.morphologyEx(gray_blur, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    norm = cv2.subtract(gray_blur, bg)

    bin_img = cv2.adaptiveThreshold(
        norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 10
    )

    white_ratio = bin_img.mean() / 255.0
    if white_ratio > 0.5: 
        bin_img = cv2.bitwise_not(bin_img)

    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cropped = bin_img[y:y+h, x:x+w]
    else:
        cropped = bin_img

    h, w = cropped.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)  # black background
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0+h, x0:x0+w] = cropped
    out = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(out).float().div(255.0).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    x = transforms.Normalize(MNIST_MEAN, MNIST_STD)(x)
    return x

# ========= Multi-digit helpers =========
def preprocess_digit(crop):
    """
    Preprocess a single digit crop to MNIST format (28x28, normalized).
    Assumes input is grayscale with white digit on black background.
    """
    # Ensure grayscale
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Pad to square
    h, w = crop.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = crop
    
    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0,1] and apply MNIST normalization
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.1307) / 0.3081  # MNIST mean and std
    return tensor

def detect_digits(image_path, model_path, min_area=50, confidence_threshold=0.5):
    """
    Detect digits in an image (white digits on black background).
    
    Args:
        image_path (str): Path to input image
        model_path (str): Path to trained model (.pth file)
        min_area (int): Minimum contour area to consider as digit
        confidence_threshold (float): Minimum confidence for prediction
    
    Returns:
        str: Detected digit sequence
        list: List of (digit, confidence, bbox) tuples
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing image: {image_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Threshold to ensure binary (white digits on black background)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if area < min_area:
            continue
        
        crop = binary[y:y+h, x:x+w]
        
        # Preprocess
        tensor = preprocess_digit(crop)
        
        # Predict
        with torch.no_grad():
            output = model(tensor.to(device))
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        confidence_val = confidence.item()
        digit = predicted.item()
        
        # Filter by confidence
        if confidence_val >= confidence_threshold:
            detections.append((digit, confidence_val, (x, y, w, h)))
    
    # Sort detections by x-coordinate (left to right)
    detections.sort(key=lambda d: d[2][0])
    
    # Extract sequence
    digit_sequence = ''.join([str(d[0]) for d in detections])
    
    print(f"Detected sequence: {digit_sequence}")
    for i, (digit, conf, bbox) in enumerate(detections):
        print(f"  Digit {i+1}: {digit} (conf: {conf:.3f}, bbox: {bbox})")
    
    return digit_sequence, detections

def visualize_detections(image_path, detections, output_path=None):
    """
    Visualize detections on the image.
    
    Args:
        image_path (str): Path to original image
        detections (list): List of (digit, confidence, bbox) tuples
        output_path (str): Path to save visualized image (optional)
    """
    # Always read as color (3-channel BGR) to ensure compatibility with drawing and saving
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    for digit, conf, (x, y, w, h) in detections:
        # Draw bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw label
        label = f"{digit}:{conf:.2f}"
        cv2.putText(img, label, (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if output_path:
        success = cv2.imwrite(output_path, img)
        if success:
            print(f"Saved visualization to: {output_path}")
        else:
            print(f"Failed to save visualization to: {output_path} (check path and permissions)")
    
# ========= Single-digit predict =========
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = MNISTCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = preprocess_mnist_opencv(args.predict).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    print("Predicted digit:", pred)
    print("Class probabilities (0..9):", np.round(probs, 4))

def export_onnx(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = MNISTCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.zeros(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model, dummy, args.export_onnx,
        input_names=["input"], output_names=["logits"],
        opset_version=13, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"Exported ONNX to {args.export_onnx}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="checkpoints/mnist_cnn.pt")
    parser.add_argument("--model", type=str, default="checkpoints/mnist_cnn.pt")
    parser.add_argument("--predict", type=str, help="Path tới ảnh ngoài để dự đoán 1 chữ số")
    parser.add_argument("--detect-digits", type=str, help="Ảnh chứa nhiều chữ số (dùng logic đơn giản)")
    parser.add_argument("--vis-out", type=str, help="Lưu ảnh trực quan hoá bbox & nhãn")
    parser.add_argument("--export-onnx", type=str, help="Xuất onnx tới file này")
    args = parser.parse_args()

    if args.detect_digits:
        # Gọi hàm mới
        sequence, detections = detect_digits(args.detect_digits, args.model)
        if args.vis_out:
            visualize_detections(args.detect_digits, detections, args.vis_out)
    elif args.predict:
        predict(args)
    elif args.export_onnx:
        export_onnx(args)
    else:
        train(args)

if __name__ == "__main__":
    main()

# python detect_number.py --detect-digits test.png --model models/mnist_cnn_pytorch.pth --vis-out visualization.jpg
