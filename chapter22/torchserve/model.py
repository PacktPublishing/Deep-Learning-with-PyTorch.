"""
Author: Atif Adib

Contains the model for MNIST classification which will be deployed for inference

"""
import os
import torch
from torchvision import transforms
from PIL import Image

class NN(torch.nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
        torch.nn.LogSoftmax(dim=1)
    )
    self.my_transforms = transforms.Compose([
    								transforms.Grayscale(),
    								transforms.Resize(size=(28,28)),
    								transforms.ToTensor(),
                                    transforms.Normalize(mean=0.0, std=1.0),
                                    transforms.Lambda(lambda x: torch.flatten(x))
                                    ])

  def forward(self, x):
    return self.model(x)

  def save_to_path(self, model_checkpoint_path):
  	torch.save(self.state_dict(), model_checkpoint_path)

  def load_from_path(self, model_checkpoint_path):
  	self.load_state_dict(torch.load(model_checkpoint_path))

  def inference(self, input_pil_image):
  	image_tensor = self.my_transforms(input_pil_image)
  	image_tensor = image_tensor.unsqueeze(0)
  	print(image_tensor.shape)
  	return self(image_tensor)
  	

