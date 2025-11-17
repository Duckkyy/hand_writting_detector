import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
import json
import re

def vlm_json_to_dict(vlm_output: str):
    text = vlm_output.strip()

    if text.startswith("```"):
        parts = re.split(r"```(?:json|JSON)?", text)
        # parts = ["", "{...}", ""]
        for p in parts:
            if "{" in p and "}" in p:
                text = p.strip()
                break
    # Extract JSON part
    lines = text.split("\n")
    clean_lines = []
    start = False
    for line in lines:
        if "{" in line:
            start = True
        if start:
            clean_lines.append(line)
        if "}" in line:
            pass

    clean_text = "\n".join(clean_lines).strip()

    # Parse JSON
    return json.loads(clean_text)


def read_table_vlm(image_path):
    import google.generativeai as gen
    from PIL import Image

    gen.configure(api_key="AIzaSyD-HLbvD45nXcVX5wLH1fUUtDoL20IhKo0")

    img = Image.open(image_path)

    model = gen.GenerativeModel("gemini-2.5-pro")  # or "gemini-1.5-pro"

    prompt = """
    You are an OCR parser.
    Read the table in the image and output the content as JSON.
    Rows = table rows, columns = table columns.

    Return JSON in this format:
    {
    "rows": [
        ["cell_0_0", "cell_0_1", ...],
        ["cell_1_0", "cell_1_1", ...]
    ]
    "crossed_out_printed_cells": [
        {"row_index": r, "col_index": c},
        ...
      ]
    }

    IMPORTANT: 
    - Only output valid JSON, nothing else.
    - Do not explain.
    """

    response = model.generate_content([prompt, img])

    data = vlm_json_to_dict(response.text)

    # print(data)
    return data

def read_table_paddleocr(image_path: str):

    ocr = PaddleOCR(
        lang="japan",             
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # device="cpu",
    )

    results = ocr.predict(image_path)
    if not results:
        print("No text detected.")
        return []

    res = results[0]
    data = res.json["res"]          

    texts  = data["rec_texts"]      
    scores = data["rec_scores"]  
    polys  = data["rec_polys"]    

    entries = []
    heights = []

    for text, score, poly in zip(texts, scores, polys):
        poly = np.array(poly)       # (4,2)
        xs = poly[:, 0]
        ys = poly[:, 1]

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        h_box = y_max - y_min

        heights.append(h_box)
        entries.append({
            "text": text,
            "score": float(score),
            "cx": cx,
            "cy": cy,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        })

    if not entries:
        print("No text detected after filtering.")
        return []

    entries = sorted(entries, key=lambda e: e["cy"])
    avg_h = float(np.mean(heights))
    row_thresh = avg_h * 0.7        
    rows = []
    current_row = []
    last_cy = None

    for e in entries:
        if last_cy is None:
            current_row = [e]
            last_cy = e["cy"]
            continue

        if abs(e["cy"] - last_cy) < row_thresh:
            current_row.append(e)
            last_cy = (last_cy * (len(current_row)-1) + e["cy"]) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [e]
            last_cy = e["cy"]

    if current_row:
        rows.append(current_row)

    row_text_lists = []
    for row in rows:
        row_sorted = sorted(row, key=lambda e: e["cx"])
        texts_in_row = [e["text"] for e in row_sorted]
        row_text_lists.append(texts_in_row)

    print("===== ROWS (grouped by rows) =====")
    for i, r in enumerate(row_text_lists):
        print(f"Row {i}: {r}")
        print(len(r))

    return row_text_lists

# read_table_vlm("image.png")
# read_table_paddleocr("image.png")