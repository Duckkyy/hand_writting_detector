import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import argparse

def run_yolo_inference(model_path, image_path, conf_threshold=0.5, iou_threshold=0.6, save_result=True):
    """
    Run YOLOv8 inference on image
    
    Args:
        model_path (str): Path to YOLOv8 model file (models/best.pt)
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold for detections
        save_result (bool): Whether to save result image
    
    Returns:
        results: YOLO detection results
        annotated_img: Image with bounding boxes drawn
    """
    
    # Load YOLOv8 model
    try:
        model = YOLO(model_path)
        print(f"Loaded YOLOv8 model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return None, None
    
    print(f"Processing image: {image_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Run inference
    results = model(image_path, conf=conf_threshold, iou=iou_threshold,verbose=False)
    
    # Get the first result (since we're processing one image)
    result = results[0]
    
    # Extract detection information
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        print(f"Found {len(boxes)} detections")
        
        # Print detection details
        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name (if available)
            class_name = model.names[class_id] if hasattr(model, 'names') else f"Class_{class_id}"
            
            print(f"  Detection {i+1}: {class_name} - conf: {confidence:.3f} - bbox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    else:
        print("No detections found")
    
    # Get annotated image
    annotated_img = result.plot()
    
    # Save result if requested
    if save_result:
        save_dir = "yolo_results"
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(save_dir, f"{base_name}_yolo_result.jpg")
        cv2.imwrite(output_path, annotated_img)
        print(f"Saved result: {output_path}")
    
    return results, annotated_img

def crop_detections(results, original_image_path, padding=5, save_crops=True):
    """
    Crop detected bounding boxes and save them to class-specific folders
    
    Args:
        results: YOLO detection results
        original_image_path (str): Path to original image
        padding (int): Padding around bounding box in pixels
    
    Returns:
        dict: Dictionary with crop information
    """
    
    if not results or len(results) == 0:
        print("No results to crop")
        return {}
    
    # Load original image
    img = cv2.imread(original_image_path)
    if img is None:
        print(f"Cannot read image: {original_image_path}")
        return {}
    
    h, w = img.shape[:2]
    result = results[0]
    boxes = result.boxes
    
    if boxes is None or len(boxes) == 0:
        print("No detections to crop")
        return {}
    
    # Get model for class names
    model_path = "models/best.pt"
    try:
        model = YOLO(model_path)
        class_names = model.names
    except:
        class_names = {}
    
    crop_info = {}
    
    print(f"Cropping {len(boxes)} detections...")
    
    for i, box in enumerate(boxes):
        # Get coordinates and class info
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        
        # Get class name
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        # Add padding to bounding box
        x1_pad = max(0, int(x1) - padding)
        y1_pad = max(0, int(y1) - padding)
        x2_pad = min(w, int(x2) + padding)
        y2_pad = min(h, int(y2) + padding)
        
        # Crop the region
        crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size == 0:
            print(f"Invalid crop for detection {i+1}")
            continue
        
        if save_crops:
            # Create class-specific directory
            output_dir="crops"
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate filename
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            crop_filename = f"{base_name}_det{i+1}_conf{confidence:.2f}.png"
            crop_path = os.path.join(class_dir, crop_filename)
        
            # Save crop
            cv2.imwrite(crop_path, crop)
            print(f"Saved crop {i+1}: {class_name} (conf: {confidence:.3f}) -> {crop_path}")
        else:
            crop_path = None
        
        # Store crop information
        crop_info[i] = {
            'class_name': class_name,
            'class_id': class_id,
            'confidence': confidence,
            'original_bbox': [int(x1), int(y1), int(x2), int(y2)],
            'padded_bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
            'crop_path': crop_path,
            'crop_size': [x2_pad - x1_pad, y2_pad - y1_pad],
            'crop_image': crop,
        }
        
    # print(f"\nCrops saved to: {output_dir}/")
    print(f"Total crops: {len(crop_info)}")
    
    # Print summary by class
    class_counts = {}
    for info in crop_info.values():
        class_name = info['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Crops by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} crops")
    
    return crop_info

def run_yolo_on_multiple_images(model_path, image_dir, conf_threshold=0.25):
    """
    Run YOLOv8 inference on multiple images in a directory
    """
    model = YOLO(model_path)
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Find all images in directory
    image_files = []
    for ext in extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Processing {len(image_files)} images from {image_dir}")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\nProcessing: {img_file}")
        
        try:
            results, annotated_img = run_yolo_inference(model_path, img_path, conf_threshold)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference and Cropping')
    parser.add_argument('--model', type=str, default='models/best.pt', 
                       help='Path to YOLOv8 model file')
    parser.add_argument('--image', type=str,
                       help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold')
    parser.add_argument('--dir', type=str,
                       help='Process all images in directory')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save result images')
    parser.add_argument('--crop', action='store_true',
                       help='Crop detected objects to class folders')
    parser.add_argument('--crop-dir', type=str, default='crops',
                       help='Directory to save crops')
    parser.add_argument('--padding', type=int, default=0,
                       help='Padding around crops in pixels')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please make sure the best.pt file exists in the current directory")
        return
    
    if args.dir:
        # Process directory
        run_yolo_on_multiple_images(args.model, args.dir, args.conf, args.crop)
    elif args.image:
        # Process single image
        results, annotated_img = run_yolo_inference(
            args.model, args.image, args.conf, args.iou, save_result=not args.no_save
        )
        
        # Crop detections if requested
        if results and args.crop:
            crop_info = crop_detections(
                results, args.image, 
                padding=args.padding,
            )
    else:
        print("Please specify either --image or --dir")

if __name__ == '__main__':
    # Example usage if run without arguments
    if len(os.sys.argv) == 1:
        print("YOLOv8 Inference and Cropping Script")
        print("\nUsage examples:")
        print("  python yolo_detection.py --image path/to/image.jpg")
        print("  python yolo_detection.py --image path/to/image.jpg --crop")
        print("  python yolo_detection.py --image path/to/image.jpg --crop --padding 20")
        print("  python yolo_detection.py --dir path/to/images/ --crop")
        print("\nParameters:")
        print("  --model: Path to model file (default: best.pt)")
        print("  --image: Input image path")
        print("  --conf: Confidence threshold (default: 0.25)")
        print("  --dir: Process all images in directory")
        print("  --no-save: Don't save annotated result images")
        print("  --crop: Crop detected objects to class-specific folders")
        print("  --padding: Padding around crops in pixels (default: 10)")
    else:
        main()

# python yolo_detection.py --image image.png --conf 0.5 --iou 0.6 --crop