import json
import os
import torch
import matplotlib.pyplot as plt
from src.models.model import CNN
from src.utils.train import train_model
from src.datasets.dataset import load_datasets

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

with open('train_params.json', 'r') as file:
    p = json.load(file)
with open('model_params.json', 'r') as file:
    p.update(json.load(file))

p["precision"] = eval(p["precision"])

train_loader, val_loader, _ = load_datasets(batch_size = p["batch_size"])

model = CNN(p['in_channels'], p['out_channels'], p['kernel_size'], p['in_height'], p['in_width'], p['out_size']).to(device)
p["loss_fn"] = eval(p['loss_fn'])
if p["optimizer"] == 'Adam':
    p["optimizer"] = torch.optim.Adam(model.parameters(), lr=p['lr'], weight_decay=1e-4)

history = train_model(model, n_epochs=p['n_epochs'], train_loader=train_loader,test_loader=val_loader, loss_fn=p['loss_fn'], optimizer=p['optimizer'], model_path="./weights/", model_name="model.pt", precision=device)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
x = range(p['n_epochs'])
y_train_loss = history['train_loss']
y_test_loss = history['test_loss']
y_train_accuracy = history['train_accuracy']
y_test_accuracy = history['test_accuracy']

ax[0].plot(x, y_train_loss, '-r', label='Train Loss')
ax[0].plot(x, y_test_loss, '-b', label='Test Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].grid()
ax[0].legend()

ax[1].plot(x, y_train_accuracy, '-g', label='Train Accuracy')
ax[1].plot(x, y_test_accuracy, '-b', label='Test Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy [%]')
ax[1].grid()
ax[1].legend()

figures_dir = './figures'
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, 'accuracy.png'), format='png')
plt.show()