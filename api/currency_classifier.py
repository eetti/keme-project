import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import numpy as np
from typing import Dict, Tuple, List

class CurrencyClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.class_names = self._get_class_names()
        self.transform = self._get_transforms()
        
    def _load_model(self) -> nn.Module:
        """Load a pre-trained ResNet model and modify for our currency classes"""
        # Use ResNet18 as base model
        model = models.resnet18(pretrained=True)
        
        # Freeze all layers except the final layer
        for param in model.parameters():
            param.requires_grad = False
            
        # Modify the final layer for our currency classes
        num_classes = len(self._get_class_names())
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load to device
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_class_names(self) -> List[str]:
        """Define currency classes for UK, US, and France"""
        return [
            "UK_5", "UK_10", "UK_20", "UK_50",
            "US_1", "US_5", "US_10", "US_20", "US_50", "US_100",
            "France_5", "France_10", "France_20", "France_50", "France_100", "France_200", "France_500"
        ]
    
    def _get_transforms(self):
        """Define image transformations for the model"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to tensor"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        """Predict currency from image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_bytes)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get class name and parse country/amount
            class_name = self.class_names[predicted_class]
            country, amount = self._parse_class_name(class_name)
            
            return {
                "country": country,
                "amount": amount,
                "confidence": confidence,
                "class_name": class_name
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return fallback prediction
            return {
                "country": "US",
                "amount": "10.00",
                "confidence": 0.0,
                "class_name": "US_10"
            }
    
    def _parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """Parse class name to extract country and amount"""
        parts = class_name.split("_")
        country = parts[0]
        amount = parts[1] + ".00"
        
        return country, amount
    
    def get_currency_code(self, country: str) -> str:
        """Get currency code for country"""
        currency_map = {
            "UK": "GBP",
            "US": "USD",
            "France": "EUR"
        }
        return currency_map.get(country, "USD")
    
    def get_exchange_rate(self, currency: str) -> float:
        """Get mock exchange rate to USD"""
        rates = {
            "GBP": 1.25,
            "USD": 1.0,
            "EUR": 1.10
        }
        return rates.get(currency, 1.0)

# Global classifier instance
classifier = CurrencyClassifier() 