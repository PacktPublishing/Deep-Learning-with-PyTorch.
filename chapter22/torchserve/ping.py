import requests
import json
import base64

# Load the JPEG image
image_path = "sample.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Encode image bytes as base64 string
base64_image = base64.b64encode(image_bytes).decode("utf-8")

# Prepare data payload with base64-encoded image
data = {"input": base64_image}

# Define TorchServe endpoint URL
url = "http://127.0.0.1:8080/predictions/mnist_model"  # Replace with your model's endpoint

# Send HTTP POST request with JSON payload
response = requests.post(url, json=data)

# Process the response
result = json.loads(response.text)
print("Model Prediction:", json.dumps(result, indent=4))