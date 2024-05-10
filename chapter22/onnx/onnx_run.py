import onnxruntime
import torch
from torchvision import transforms
from PIL import Image

# Transformation Pipeline
my_transforms = transforms.Compose([
							transforms.Grayscale(), transforms.Resize(size=(28,28)),
							transforms.ToTensor(), transforms.Normalize(mean=0.0, std=1.0),
		                    transforms.Lambda(lambda x: torch.flatten(x))
		                    ])

# Load the ONNX model
onnx_model = 'mnist_model.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model)

# Example input (adjust according to your model's input shape)
pil_image = Image.open('sample.jpg')
input_data = my_transforms(pil_image).unsqueeze(0)
input_data = input_data.numpy()

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: input_data}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)  # This will show the model's output

