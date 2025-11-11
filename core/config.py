# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL= str(os.getenv("DATABASE_URL"))
    REDIS_URL = str(os.getenv("REDIS_URL", "redis://localhost:6379"))
    ENV= str(os.getenv("ENV", "dev"))
    DEBUG = bool(os.getenv("DEBUG", "true").lower() == "true")
    PORT = int(os.getenv("PORT", 8000))

settings = Settings()