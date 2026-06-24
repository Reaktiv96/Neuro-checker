"""Configuration for the application."""
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    """Application configuration."""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    RUBLES_IN_DOLLAR = float(os.getenv("RUBLES_IN_DOLLAR", "0"))
    RUB_PER_TOKEN = float(os.getenv("RUB_PER_TOKEN", "0.0005"))
    EXTERNAL_SERVICE_URL = os.getenv(
        "EXTERNAL_SERVICE_URL", 
        "http://62.113.108.33/platform-v1/solving-dz"
    )
    EXTERNAL_SERVICE_AUTH = os.getenv(
        "EXTERNAL_SERVICE_AUTH",
        "b59210ae-1493-46c6-b37b-8e89ffa86d90"
    )
    AUTH_EMAIL = os.getenv("AUTH_EMAIL", "artem.kashtalap@gmail.com").strip().lower()
    AUTH_PASSWORD = os.getenv("PASSWORD", "").strip()
    STUDENT_NAME = os.getenv("STUDENT_NAME", "Студент")
    SECRET_KEY = os.getenv("SECRET_KEY", "")

    _ENV_NAME = os.getenv("FLASK_ENV", "production").strip().lower()
    _DEFAULT_SECURE_COOKIE = "false" if _ENV_NAME == "development" else "true"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", _DEFAULT_SECURE_COOKIE).strip().lower() == "true"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_DURATION_DAYS = int(os.getenv("SESSION_DURATION_DAYS", "7"))
    PERMANENT_SESSION_LIFETIME = timedelta(days=SESSION_DURATION_DAYS)

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
