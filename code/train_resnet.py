import os
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class ASLLetterDataSet(Dataset):
    def __init__(self, data_dir="../data/train/images/") -> None:
        super().__init__()
        # Gets file directory
        self.data_dir = data_dir
        self.file_list = [filename for filename in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, filename))]
    
    def __len__(self):
        # The length of the number of data (number of images)
        data_length = len(self.file_list)
        return data_length
    
    def __getitem__(self, index):
        # Gets label
        filename = self.file_list[index]
        label = ord(filename[0]) - 65
        
        # Opens image
        image = Image.open(self.data_dir + filename)
        
        # Scale the image for 384 to 224.
        image = image.resize((224, 224))
        
        # Convert image to numpy array
        image = np.asarray(image)
        
        # Opens image and scales it to 0 - 1 RGB
        image = image / 255.0
        
        # Normalizing Image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Transposing for pytorch. (rgb, width, height)
        image = np.transpose(image, (2, 0, 1))
        
        return image, label


if __name__ == '__main__':
    
    # device = "cpu"
    device = "cuda"
    
    asl_training = ASLLetterDataSet()
    asl_train_dataloader = DataLoader(asl_training, batch_size=32, shuffle=True, num_workers=2)
    
    asl_valid = ASLLetterDataSet(data_dir="../data/valid/images/")
    asl_valid_dataloader = DataLoader(asl_valid, batch_size=32, shuffle=False, num_workers=2)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # Change last layer of resnet from 100 option output to the required 26 (alphabet).
    model.fc = torch.nn.Linear(in_features=512, out_features=26)

    # Enables traning parameters.
    model = model.to(device)
    
    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    def train_epoch():
        model.train()
        losses = []
        
        for data in tqdm(asl_train_dataloader):
            images, labels = data
            
            images = images.float()
            labels =  labels.long()
            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(output, labels)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
        
        avg_losses = sum(losses) / len(losses)
        
        return avg_losses
 
 
    def valid_epoch():
        model.eval()
        losses = []
        
        for data in tqdm(asl_valid_dataloader):
            images, labels = data
            
            images = images.float()
            labels =  labels.long()
            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(output, labels)
            losses.append(loss.item())
        
        avg_losses = sum(losses) / len(losses)
        
        return avg_losses   
    
    
    
    # Training Loop
    epochs = 10
    avg_train_losses, avg_valid_losses = [], []
    
    for epoch in tqdm(range(epochs)):
        avg_training_loss = train_epoch()
        avg_train_losses.append(avg_training_loss)
        
        avg_valid_loss = valid_epoch()
        avg_valid_losses.append(avg_valid_loss)
    
    torch.save(model, "./letter_classifier_model.pt")
    
    plt.plot(avg_train_losses, color='red', label="Train_Loss")
    plt.plot(avg_valid_losses, color='blue', label="Valid Loss")
    plt.legend()
    plt.show()