#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[21]:


# 🌐 Computer Vision
import torch
from torch import nn # Neural neutwork components
from torch.utils.data import DataLoader # For batching and loading the data
from torchvision import datasets # To access FashionMNIST
from torchvision.transforms import ToTensor # Convert PIL images to tensors

# 🕒 Utilities
from timeit import default_timer as timer # Training time duration
from tqdm import tqdm
import random

# 📊 Visualization and metrics
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics import Accuracy, ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


# ## Transform data and create dataloaders

# In[22]:


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_transforms(num_channels=1):
    """
    Devuelve las transformaciones de datos para imágenes con un número definido de canales.
    
    Args:
        num_channels (int): Número de canales de salida (1 para escala de grises, 3 para RGB).
        
    Returns:
        data_transforms (torchvision.transforms.Compose): Transformaciones aplicadas a las imágenes.
    """
    grayscale_transform = transforms.Grayscale(num_output_channels=num_channels)
    normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if num_channels == 3 else ([0.485], [0.229])
    
    data_transforms = transforms.Compose([
        grayscale_transform,
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(*normalization)
    ])
    return data_transforms


def get_dataloaders(data_dir, batch_size, num_channels, transform=None):
    """
    Carga los datasets y devuelve los DataLoaders para entrenamiento y prueba.
    
    Args:
        data_dir (str): Ruta al directorio que contiene los datos.
        batch_size (int): Tamaño del batch.
        transform (torchvision.transforms.Compose, optional): Transformaciones para las imágenes.

    Returns:
        tuple: DataLoader de entrenamiento, DataLoader de prueba y nombres de las clases.
    """
    if transform is None:
        transform = get_data_transforms(num_channels)  # Llamada a la función para obtener las transformaciones por defecto
    
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    class_names = train_data.classes
    
    return train_data, test_data, train_dataloader, test_dataloader, class_names


# ## Train and Test functions

# In[23]:


def train_step(model, data_loader, loss_fn, optimizer, accuracy_metric, device=device):
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in data_loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        y = y.unsqueeze(1).float()  # Asegurar que las etiquetas tengan tamaño [batch_size, 1]
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda"):  # Mixed Precision
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        preds = (torch.sigmoid(y_pred) > 0.5).int()  # Convertir logits a predicciones binarias
        train_acc += accuracy_metric(preds, y.int())

    return train_loss / len(data_loader), train_acc.item() / len(data_loader)



def test_step(model, data_loader, loss_fn, accuracy_metric, device=device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            y = y.unsqueeze(1).float()  # Asegurar que las etiquetas tengan tamaño [batch_size, 1]
            with torch.amp.autocast(device_type="cuda"):  # Mixed Precision
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
            
            test_loss += loss.item()
            preds = (torch.sigmoid(y_pred) > 0.5).int()  # Convertir logits a predicciones binarias
            test_acc += accuracy_metric(preds, y.int())

    return test_loss / len(data_loader), test_acc.item() / len(data_loader)


# ## Results

# In[24]:


def visualize_predictions(model, test_data, test_dataloader, class_names, device):
    """
    Visualiza predicciones de un modelo en muestras aleatorias del conjunto de datos de prueba.

    Args:
        model (torch.nn.Module): El modelo entrenado.
        test_data (torch.utils.data.Dataset): Dataset de prueba.
        test_dataloader (torch.utils.data.DataLoader): DataLoader del dataset de prueba.
        class_names (list): Lista con los nombres de las clases.
        device (torch.device): Dispositivo donde está el modelo (CPU o GPU).
    """
    model.eval()
    with torch.inference_mode():
        y_pred_tensor = torch.cat([model(X.to(device)).argmax(dim=1).cpu() for X, _ in test_dataloader])

    # Seleccionar 16 muestras aleatorias
    test_samples = random.sample(list(test_data), k=16)
    test_labels = [label for _, label in test_samples]

    # Predicciones para cada muestra
    pred_probs = [model(torch.unsqueeze(sample, dim=0).to(device)).softmax(dim=1).cpu() for sample, _ in test_samples]
    pred_classes = [prob.argmax() for prob in pred_probs]

    # Visualización
    plt.figure(figsize=(5, 5))
    for i, (sample, label) in enumerate(test_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(sample[0], cmap="gray")
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[label]
        plt.title(f"Pred: {pred_label}\nTruth: {truth_label}", fontsize=10,
                  color="g" if pred_label == truth_label else "r")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

