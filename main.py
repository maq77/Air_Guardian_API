from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import forecast, health, policy
from config.settings import settings

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast.router)
app.include_router(health.router)
app.include_router(policy.router)

@app.get("/")
def root():
    return {
        "message": "AirGuardian API",
        "version": "1.0.0",
        "endpoints": {
            "forecast": "/api/v1/forecast/{location}",
            "health": "/api/v1/health-alert/{location}",
            "policy": "/api/v1/policy-recommendations/{location}",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)