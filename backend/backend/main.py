from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.database import engine, Base
from backend.routes import auth_routes

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Insomnia Risk Predictor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_routes.router)

@app.get("/")
async def root():
    return {"message": "Insomnia Risk Predictor API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
