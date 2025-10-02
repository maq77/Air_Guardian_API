import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from models.forecaster import AQIForecaster

MODEL_PATHS = {
    1: "trained_models/forecaster_1h_multi_city.pkl",
    6: "trained_models/forecaster_6h_multi_city.pkl",
    24: "trained_models/forecaster_24h_multi_city.pkl"
}

loaded_models = {}

def load_models():
    for hours, path in MODEL_PATHS.items():
        if os.path.exists(path):
            print(f"Loading model for {hours}h from {path}")
            forecaster = AQIForecaster(forecast_hours=hours)
            forecaster.load(path)
            loaded_models[hours] = forecaster
        else:
            print(f"Model file not found for {hours}h: {path}")

load_models()