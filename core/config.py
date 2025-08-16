from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Loads environment variables from the .env file."""
    GOOGLE_API_KEY: str
    LLAMA_CLOUD_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str

    # Load from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create a single instance to be used across the application
settings = Settings()