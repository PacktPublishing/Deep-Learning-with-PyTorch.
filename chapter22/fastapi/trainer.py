"""
 Train the MNIST Neural Network for Classification

"""
import torch
from model import NN
from dataset import get_dataset

def train(model, data, device, criterion, optimizer, epochs=10):
  for epoch in range(epochs):
    total_loss = 0.0
    count = 0.0
    for inputs, labels in data:
      inputs = inputs.to(device)
      labels = labels.to(device)
      output = model(inputs)
      loss = criterion(output, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      count += 1

    print(f"Epoch:{epoch}| Loss: {total_loss / count}")

def eval(model, data, device):
  total, correct = 0, 0
  for inputs, labels in data:
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model(inputs)
    output = output.argmax(dim=1)
    correct_predictions = (output == labels).sum()
    total_predictions = labels.size()[0]
    total += total_predictions
    correct += correct_predictions
  print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
  # Initialize Model
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = NN().to(device)

  # Setup Optimizers and Loss function
  optimizer = torch.optim.Adam(model.parameters())
  criterion = torch.nn.CrossEntropyLoss()

  # Datasets
  train_dataloader, test_dataloader = get_dataset()

  model.train()
  train(model, train_dataloader, device, criterion, optimizer, epochs=10)
  model.eval()
  eval(model, test_dataloader, device)
  torch.save(model.state_dict(), 'checkpoint.pt')
