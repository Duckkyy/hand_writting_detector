import google.generativeai as genai
from PIL import Image

genai.configure(api_key="AIzaSyD-HLbvD45nXcVX5wLH1fUUtDoL20IhKo0")

model = genai.GenerativeModel("gemini-2.5-pro")  # hoặc "gemini-1.5-pro"

# img = Image.open("res1.png")
img = Image.open("crops/number/ocr_det4_conf0.64.png")

# prompt = "Read the handwritten numbers in this image and return them as a numeric string. Respond with only the number and nothing else."
prompt = "Read the red numbers in this image and return them as a numeric string. Respond with only the number and nothing else."

response = model.generate_content([prompt, img])

print(response.text)
