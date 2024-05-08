import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataset():
	# Load Training data
	training_data = datasets.MNIST(
	    root="data",
	    train=True,
	    download=True,
	    transform=transforms.Compose([transforms.ToTensor(),
	                                  transforms.Normalize(mean=0.0, std=1.0),
	                                  transforms.Lambda(lambda x: torch.flatten(x))])
	)

	test_data = datasets.MNIST(
	    root="data",
	    train=False,
	    download=True,
	    transform=transforms.Compose([transforms.ToTensor(),
	                                  transforms.Normalize(mean=0.0, std=1.0),
	                                  transforms.Lambda(lambda x: torch.flatten(x))])
	)

	train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
	return train_dataloader, test_dataloader