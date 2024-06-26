import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

# import neural network class from our model.py file 
from model import NN

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NN().to(device)
model.load_from_path('./checkpoint.pt')


@app.post("/mnist/v1/")
async def predict(image: UploadFile = File(...)):
	contents = await image.read()
	pil_image = Image.open(BytesIO(contents))
	output = model.inference(pil_image)

	# Get Class
	class_name = f"Digit-{output.argmax()}"

	# Get Probabilities
	probs = torch.exp(output)

	# Normalize probabilities across the second dimension (dim=1)
	probs_normalized = probs / torch.sum(probs, dim=1, keepdim=True)

	return {"output": {"probabilities": np.round(probs.detach().numpy(), 4).tolist(), "prediction": class_name}}

if __name__ == '__main__':
	uvicorn.run(app, host="0.0.0.0", port=9000)
