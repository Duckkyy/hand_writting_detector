from yolo_detection import run_yolo_inference, crop_detections
from get_red_mask import get_red_mask
from detect_number import detect_digits, visualize_detections
from read_table import read_table_vlm
import cv2
import os
import numpy as np
import json

def detect_handwritten_numbers(image_path, yolo_model_path, digit_model_path):
    # Step 1: Detect handwritten regions using YOLO
    detections, _ = run_yolo_inference(yolo_model_path, image_path, save_result=True)

    crop_images = crop_detections(detections, image_path, save_crops=True)

    fixed_sequences = []

    # Step 2: For each detected region, extract red ink and recognize digits
    for crop_image in crop_images:
        if crop_images[crop_image]["class_name"] != "number": continue
        _, _, mask_path = get_red_mask(crop_images[crop_image]["crop_image"], save_result=True)

        digit_sequence, detections = detect_digits(mask_path, digit_model_path)

        fixed_sequences.append(digit_sequence)

    print(fixed_sequences)
    table_data, fixed_cells = retrieve_table_data(image_path)

    for i in range(len(fixed_cells)):
        r, c = fixed_cells[i]
        table_data[r][c] = fixed_sequences[i]

    return table_data

def retrieve_table_data(image_path):
    # Step 3: Read table data with handwritten corrections using VLM
    image_data = read_table_vlm(image_path)

    table_data = image_data["rows"]
    fixed_data = image_data["crossed_out_printed_cells"]

    fixed_cells = []
    for cell in fixed_data:
        r, c = cell["row_index"], cell["col_index"]
        # print(f"Cell at row {r}, column {c} has handwritten correction: {table_data[r][c]}")
        fixed_cells.append((r,c))

    return table_data, fixed_cells

def save_data_json(output_path="output.json"):
    new_table = detect_handwritten_numbers(
        image_path="image.png",
        yolo_model_path="models/best.pt",
        digit_model_path="models/mnist_cnn_pytorch.pth"
    )
    with open(output_path, "w") as f:
        json.dump(new_table, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {output_path}")

def run_pipeline():
    save_data_json()

if __name__ == "__main__":
    run_pipeline()