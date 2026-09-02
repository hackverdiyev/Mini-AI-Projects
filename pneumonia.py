import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir="./Data/Pneumonia/train"
test_dir="./Data/Pneumonia/test"

transform_train=transforms.Compose([ 
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_test=transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset=datasets.ImageFolder(train_dir,transform=transform_train)
test_dataset=datasets.ImageFolder(test_dir,transform=transform_test)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

print("Data Loaded")


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN,self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*18*18,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x=self.conv_layers(x)
        x=self.fc_layers(x)
        return x


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=PneumoniaCNN().to(device)

print("CNN created")


criterion=nn.BCELoss()  
optimizer=optim.Adam(model.parameters(), lr=0.001)

def train_model(model,train_loader,test_loader,criterion,optimizer,epochs):
    train_loss,test_loss,test_acc=[],[],[]
    for epoch in range(epochs):
        model.train()
        epoch_loss=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.float().to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs.squeeze(),labels)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        train_loss.append(epoch_loss/len(train_loader))
        
        model.eval()
        correct,total,test_epoch_loss=0,0,0
        with torch.no_grad():
            for images,labels in test_loader:
                images,labels=images.to(device),labels.float().to(device)
                outputs=model(images)
                test_loss_batch=criterion(outputs.squeeze(),labels)
                test_epoch_loss+=test_loss_batch.item()
                preds=(outputs.squeeze()>0.5).float()
                correct+=(preds==labels).sum().item()
                total+=labels.size(0)
        val_loss.append(val_epoch_loss/len(val_loader))
        val_acc.append(correct/total)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, "
              f"Val Loss: {test_loss[-1]:.4f}, Val Acc: {test_acc[-1]*100:.2f}%")
    
    return train_loss,test_loss,test_acc

train_loss,test_loss,test_acc=train_model(model,train_loader,test_loader,criterion,optimizer,epochs=50)

