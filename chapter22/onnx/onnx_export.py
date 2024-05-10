import torch
from model import NN

# Load the saved model
model = NN()
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()  # Set the model to inference mode

# Example input (adjust according to your model's input shape)
dummy_input = torch.randn(1, 784)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, 'mnist_model.onnx', verbose=True)

