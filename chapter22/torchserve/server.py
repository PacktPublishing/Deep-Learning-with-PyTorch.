# Main Server File for TorchServe
from io import BytesIO
import base64
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from model import NN


class MNISTHandler(nn.Module):
	def initialize(self, context):
		self.model = NN()
		self.model.load_state_dict(torch.load('checkpoint.pt'))
		self.model.eval()
		self.transform = transforms.Compose([
							transforms.Grayscale(), transforms.Resize(size=(28,28)),
							transforms.ToTensor(), transforms.Normalize(mean=0.0, std=1.0),
		                    transforms.Lambda(lambda x: torch.flatten(x))
		                    ])
		self.initialized = True

	def preprocess(self, data):
		base64_image = data[0]['body']["input"]
		pil_image = Image.open(BytesIO(base64.b64decode(base64_image)))
		input_tensor = self.transform(pil_image)
		return input_tensor.unsqueeze(0)

	def inference(self, input_data):
		with torch.no_grad():
			output = self.model(input_data)
		return output

	def postprocess(self, data):
		# Get Probabilities
		probs = torch.exp(data)
		# Normalize probabilities across the second dimension (dim=1)
		probs_normalized = probs / torch.sum(probs, dim=1, keepdim=True)
		return probs_normalized.tolist()

	def handle(self, data, context):
		model_input = self.preprocess(data)
		model_output = self.inference(model_input)
		return self.postprocess(model_output)
