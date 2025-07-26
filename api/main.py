from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from banknote_net_inference import predict_banknote
from io import BytesIO

app = FastAPI()

# Allow CORS for development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions for currency and exchange rate
CURRENCY_MAP = {
    'USD': 'USD',
    'GBP': 'GBP',
    'EUR': 'EUR',
}
EXCHANGE_RATES = {
    'USD': 1527,
    'GBP': 2057,
    'EUR': 1794,
}

def parse_prediction(pred):
    # pred is like 'USD_20', 'GBP_10', etc.
    parts = pred.split('_')
    country = parts[0]
    amount = parts[1] if len(parts) > 1 else ''
    return country, amount

@app.post("/convert-currency")
async def convert_currency(file: UploadFile = File(...)):
    contents = await file.read()
    pred, confidence = predict_banknote(BytesIO(contents))
    country, amount = parse_prediction(pred)
    currency_code = CURRENCY_MAP.get(country, country)
    exchange_rate = EXCHANGE_RATES.get(currency_code, 1.0)
    try:
        original_amount = float(amount)
    except Exception:
        original_amount = 0.0
    converted_amount = original_amount * exchange_rate
    prediction = {
        "country": country,
        "amount": amount,
        "confidence": confidence
    }
    response = {
        "predicted_country": prediction["country"],
        "filename": file.filename,
        "originalCurrency": currency_code,
        "originalAmount": prediction["amount"],
        "convertedAmount": f"{converted_amount:.2f}",
        "exchangeRate": str(exchange_rate),
        "confidence": f"{prediction['confidence']:.3f}"
    }
    return JSONResponse(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 