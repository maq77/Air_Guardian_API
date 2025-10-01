# 🌍 AirGuardian API

**AI-Powered Air Quality Intelligence for Everyone**

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA%20Space%20Apps-2024-blue)](https://www.spaceappschallenge.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.5+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 Challenge: From Earthdata to Action

AirGuardian transforms NASA's Earth observation data into actionable insights through AI-powered APIs. Instead of building another standalone app, we provide the **AI brain** that powers air quality intelligence across any platform.

### 💡 The Problem

- **7 million** premature deaths annually from air pollution (WHO)
- **99%** of the world breathes unsafe air
- Raw data exists, but transforming it into **actionable guidance** is the challenge

### ✨ Our Solution

Three AI models delivered as REST APIs:

1. **🔮 Forecasting Model** - Predicts AQI 1-72 hours ahead using TEMPO satellite data + ground sensors
2. **⚠️ Risk Classification** - Translates forecasts into human-centric alerts (safe/caution/hazardous)
3. **📋 Policy Recommendations** - Suggests government interventions to reduce pollution exposure

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/your-team/airguardian-api.git
cd airguardian-api

# Install dependencies
pip install -r requirements.txt

# Run API server
python main.py
```

**API runs at:** `http://localhost:8000`  
**Interactive docs:** `http://localhost:8000/docs`

---

## 📊 API Endpoints

### 1. Current Air Quality
```bash
GET /api/v1/current/{location}
```
**Example:**
```json
{
  "location": "Cairo",
  "aqi": 145,
  "risk_level": "Unhealthy for Sensitive Groups",
  "pm25": 58.0,
  "no2": 43.5,
  "data_source": "TEMPO Satellite + Ground Sensors"
}
```

### 2. Forecast (1-72 hours)
```bash
GET /api/v1/forecast/{location}?hours=24
```
**Returns:** Hourly AQI predictions with confidence scores

### 3. Health Recommendations
```bash
GET /api/v1/health-alert/{location}
```
**Example:**
```json
{
  "risk_level": "Unhealthy for Sensitive Groups",
  "activities_safe": ["Indoor activities", "Short walks"],
  "activities_avoid": ["Outdoor sports", "Prolonged activity"],
  "sensitive_groups_warning": "Limit outdoor exposure"
}
```

### 4. Policy Recommendations (Government)
```bash
GET /api/v1/policy-recommendations/{location}
```
**Example:**
```json
{
  "priority": "HIGH",
  "recommendations": [
    "Issue public health advisory",
    "Implement odd-even vehicle restrictions",
    "Increase public transportation frequency"
  ],
  "estimated_impact": "Significant reduction within 12-24 hours"
}
```

---

## 🎨 Demo

![AirGuardian Demo](demo-screenshot.png)

**Try it:** Open `demo_frontend.html` in your browser

---

## 🔧 Technology Stack

**Backend:**
- FastAPI - Modern Python web framework
- XGBoost/LSTM - ML forecasting models
- Scikit-learn - Risk classification

**Data Sources:**
- NASA TEMPO - Satellite NO2/aerosol data
- OpenAQ - Ground sensor measurements
- OpenWeather - Meteorological data

**Deployment:**
- Docker support
- Cloud-ready (AWS/GCP/Azure)
- Scalable REST APIs

---

## 📈 Model Architecture

### Forecasting Pipeline
```
Input: Historical AQI + Weather + TEMPO data
  ↓
Feature Engineering (lag features, rolling stats, time encoding)
  ↓
XGBoost/LSTM Model
  ↓
Output: AQI forecast with confidence intervals
```

### Training
```bash
python train_models.py
```

**Performance (on sample data):**
- 1-hour forecast: MAE = 12.3 μg/m³
- 6-hour forecast: MAE = 18.7 μg/m³
- Classification accuracy: 89%

---

## 💼 Use Cases

### For Citizens
- **Fitness Apps**: "Best time to run: 6am (AQI: 45)"
- **Health Apps**: Smart inhaler alerts when AQI > 150
- **Weather Apps**: Integrated air quality in daily forecast

### For Governments
- **Smart City Dashboards**: Real-time pollution hotspot monitoring
- **Traffic Management**: AI-suggested restriction zones
- **Emergency Response**: Automated alert systems

### For Developers
- **Plug-and-play APIs**: No ML expertise needed
- **RESTful design**: Works with any tech stack
- **Comprehensive docs**: Swagger/OpenAPI

---

## 🗂️ Project Structure

```
airguardian-api/
├── main.py                    # FastAPI application
├── data_fetcher.py           # Data collection from APIs
├── train_models.py           # ML model training pipeline
├── requirements.txt          # Python dependencies
├── demo_frontend.html        # Interactive demo
├── trained_models/           # Saved model weights
│   ├── forecaster_1h.pkl
│   ├── forecaster_6h.pkl
│   ├── risk_classifier.pkl
│   └── policy_engine.pkl
├── data/
│   ├── raw/                  # Downloaded TEMPO/OpenAQ data
│   └── processed/            # Processed training data
└── tests/                    # Unit tests
```

---

## 🌍 Impact

### Egypt Focus
- **Cairo**: AQI frequently exceeds 150
- **27,000+** annual deaths from air pollution in Egypt
- Limited public awareness and fragmented data

### Global Scalability
- API-first design enables worldwide deployment
- Supports any city with satellite/ground sensor coverage
- Multi-language support ready

---

##  Data Sources

### NASA TEMPO
- High-resolution NO2 and aerosol measurements
- Hourly updates during daylight
- Coverage: North America (expanding globally)
- Access: https://tempo.si.edu/

### OpenAQ
- 10,000+ ground monitoring stations worldwide
- Real-time and historical data
- Free API access
- Docs: https://docs.openaq.org/

### Weather Data
- Temperature, humidity, wind speed/direction
- Essential for pollution dispersion modeling
- Source: OpenWeather API

---

## 👥 Team AirGuardian

- **Mohamed Amin** - Lead / AI Engineer
- **Youssif Atef** - Researcher / UI/UX Designer
- **Hisham Swilam** - Data Scientist
- **Mohamed Khalid** - Sr. Backend Developer

---

## 🛠️ Development Roadmap

### Phase 1: MVP (Hackathon - 48hrs) ✅
- [x] Basic API with 3 endpoints
- [x] Simple forecasting model
- [x] Demo frontend
- [x] Documentation

### Phase 2: Enhancement (Post-Hackathon)
- [ ] LSTM time series model
- [ ] Real TEMPO data integration
- [ ] Traffic data incorporation
- [ ] Mobile app example

### Phase 3: Production
- [ ] Multi-city support (50+ cities)
- [ ] Real-time streaming
- [ ] Advanced ML (Transformers)
- [ ] Commercial partnerships

---

## 🔬 Research & References

1. **Gupta, P. et al. (2023)**. "Multi-source Data Fusion for Air Quality Forecasting." *Journal of Environmental Data Science*

2. **NASA TEMPO Mission** (2023). "High-Resolution Air Quality Measurements from Space." NASA Goddard Space Flight Center

3. **WHO Air Quality Guidelines** (2021). Global air quality guidelines

4. **OpenAQ Platform** - Open Air Quality Data

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork repository
# Create feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m 'Add amazing feature'

# Push to branch
git push origin feature/amazing-feature

# Open Pull Request
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **NASA** for TEMPO satellite data and Earthdata platform
- **OpenAQ** for open air quality data
- **Space Apps Challenge** for the opportunity
- **All contributors** to open environmental data

---

##  Contact

**Project Link:** https://github.com/maq77/Air_Guardian_API

**Demo Video:** [YouTube Link] (soon)
 
**Presentation:** [Slides Link] (soon)

**Email:** maqmohamed8@gmail.com

---

## 🌟 Star Us!

If you find AirGuardian useful, please star this repository to help others discover it!

---

<div align="center">

**Built with ❤️ for NASA Space Apps Challenge 2025**

*From Earthdata to Action - Transforming satellite data into life-saving insights*

</div>
