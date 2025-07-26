# Currency Classification API

A FastAPI backend that uses deep learning to classify currency denominations from images of UK, US, and French banknotes.

## Features

- **Image Classification**: Uses ResNet18 with transfer learning or BankNote-Net encoder for currency classification
- **Multi-Country Support**: UK (GBP), US (USD), France (EUR)
- **Currency Conversion**: Converts detected amounts to USD using configurable exchange rates
- **Confidence Scoring**: Returns prediction confidence for each classification

## Setup

### 1. Install Dependencies

```bash
python3 -m venv tf24env
source tf24env/bin/activate
pip install tensorflow==2.11.0
```


```bash
pip install -r requirements.txt
```

### 2. Dataset Structure (for ResNet Training)

To train the ResNet-based model, organize your images in the following structure:

```
currency_dataset/
├── UK_5/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── UK_10/
│   ├── image1.jpg
│   └── ...
├── US_1/
│   ├── image1.jpg
│   └── ...
├── US_5/
│   └── ...
├── France_5/
│   └── ...
└── ...
```

**Supported Classes:**
- UK: UK_5, UK_10, UK_20, UK_50
- US: US_1, US_5, US_10, US_20, US_50, US_100
- France: France_5, France_10, France_20, France_50, France_100, France_200, France_500

### 3. Using BankNote-Net for Currency Recognition

#### **A. Download BankNote-Net Assets**
- Download the encoder model (`banknote_net_encoder.h5`) and embeddings (`banknote_net.feather`) from the [BankNote-Net GitHub repository](https://github.com/microsoft/banknote-net).
- Place them in a directory such as `api/banknote_net_assets/`.

#### **B. Train a Classifier Using BankNote-Net Embeddings**
1. Use the provided script to train a classifier (e.g., RandomForest) on the embeddings:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_feather('banknote_net_assets/banknote_net.feather')
target_currencies = ['USD', 'GBP', 'EUR']
df = df[df['currency'].isin(target_currencies)]
X = df.iloc[:, :256].values
y = df['currency'] + '_' + df['denomination'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, 'banknote_net_assets/banknote_classifier.joblib')
print('Test accuracy:', clf.score(X_test, y_test))
```

2. This will create `banknote_classifier.joblib` for use in the backend.

#### **C. Retraining the Classifier**
- If you want to support more currencies or update the classifier, repeat the above process with your updated `banknote_net.feather` or your own embeddings.
- You can also generate new embeddings for your own images using the BankNote-Net encoder and add them to the dataset before retraining.

#### **D. Integration in FastAPI Backend**
- The backend uses the encoder and classifier to process uploaded images, generate embeddings, and predict the currency and denomination.
- The `/convert-currency` endpoint will return a response with the following fields:

```json
{
  "predicted_country": "USD",
  "filename": "currency_image.jpg",
  "originalCurrency": "USD",
  "originalAmount": "20",
  "convertedAmount": "30540.00",
  "exchangeRate": "1527",
  "confidence": "0.892"
}
```

### 4. Running the API

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Usage

### POST /convert-currency

Upload an image to classify the currency and get conversion details.

**Request:**
- Method: `POST`
- URL: `http://localhost:8000/convert-currency`
- Content-Type: `multipart/form-data`
- Body: Form data with key `file` containing the image

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/convert-currency" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/currency_image.jpg"
```

**Response:**
```json
{
  "predicted_country": "USD",
  "filename": "currency_image.jpg",
  "originalCurrency": "USD",
  "originalAmount": "20",
  "convertedAmount": "30540.00",
  "exchangeRate": "1527",
  "confidence": "0.892"
}
```

## Model Architecture

- **Option 1:** ResNet18 pre-trained on ImageNet (transfer learning)
- **Option 2:** BankNote-Net encoder + shallow classifier (RandomForest, etc.)
- **Input Size:** 224x224 RGB images
- **Output:** Currency and denomination
- **Training:** See above for retraining instructions

## Exchange Rates

Exchange rates are configurable in `main.py`:
```python
EXCHANGE_RATES = {
    'USD': 1527,
    'GBP': 2057,
    'EUR': 1794,
}
```

## Development

### Adding New Currencies

1. Update the classifier training script to include new currencies/denominations.
2. Retrain the classifier and update `banknote_classifier.joblib`.
3. Update `CURRENCY_MAP` and `EXCHANGE_RATES` in `main.py` as needed.

### Improving Accuracy

1. **More Data:** Add more images/embeddings for each denomination.
2. **Data Augmentation:** Use the encoder to generate embeddings for augmented images.
3. **Model Architecture:** Try different classifiers (SVM, XGBoost, etc.).
4. **Hyperparameter Tuning:** Adjust classifier parameters.

## Troubleshooting

- **TensorFlow/Keras errors:** Ensure you are using TensorFlow 2.6+ and do not have a standalone Keras package installed.
- **Model loading errors:** Check that `banknote_net_encoder.h5` and `banknote_classifier.joblib` are present and compatible.
- **Poor accuracy:** Retrain the classifier with more or better data.

## License

This project is for educational purposes. Please ensure you have proper licenses for any currency images used in training.

For BankNote-Net, see: [BankNote-Net GitHub](https://github.com/microsoft/banknote-net) 