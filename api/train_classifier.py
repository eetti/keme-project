import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple
import json

class CurrencyDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Directory with images organized in subdirectories by class
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()
    
    def _get_classes(self) -> List[str]:
        """Get list of class names from subdirectories"""
        return sorted([d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))])
    
    def _load_images(self) -> List[Tuple[str, int]]:
        """Load image paths and their corresponding class indices"""
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    images.append((img_path, class_idx))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx

def train_model(data_dir: str, model_save_path: str = "currency_model.pth"):
    """Train the currency classifier"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = CurrencyDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize model
    from currency_classifier import CurrencyClassifier
    classifier = CurrencyClassifier()
    
    # Unfreeze some layers for fine-tuning
    for param in classifier.model.layer4.parameters():
        param.requires_grad = True
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': classifier.model.fc.parameters(), 'lr': 0.001},
        {'params': classifier.model.layer4.parameters(), 'lr': 0.0001}
    ], lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    device = classifier.device
    
    print(f"Training on {len(dataset)} images with {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")
    
    for epoch in range(num_epochs):
        classifier.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100 * correct/total:.2f}%')
        
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    # Save the trained model
    torch.save(classifier.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save class mapping
    class_mapping = {idx: cls for idx, cls in enumerate(dataset.classes)}
    with open("class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print("Class mapping saved to class_mapping.json")

if __name__ == "__main__":
    # Example usage
    data_dir = "currency_dataset"  # Directory with subdirectories for each class
    train_model(data_dir) 