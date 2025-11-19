"""
Configuration management for the application
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class OpenAIConfig(BaseModel):
    """OpenAI configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))


class MilvusConfig(BaseModel):
    """Milvus/Zilliz Cloud configuration"""
    uri: Optional[str] = Field(default_factory=lambda: os.getenv("MILVUS_URI"))
    token: Optional[str] = Field(default_factory=lambda: os.getenv("MILVUS_TOKEN"))
    host: str = Field(default_factory=lambda: os.getenv("MILVUS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("MILVUS_PORT", "19530")))
    collection_name: str = "factsheet_chunks"
    
    @property
    def is_cloud(self) -> bool:
        """Check if using Zilliz Cloud"""
        return self.uri is not None and self.token is not None


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    dimension: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "384")))
    device: str = "cpu"  # Change to "cuda" if GPU available


class AppConfig(BaseModel):
    """Main application configuration"""
    title: str = Field(default_factory=lambda: os.getenv("APP_TITLE", "Bajaj AMC Factsheet Chatbot"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG_MODE", "False").lower() == "true")
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Sub-configurations
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


# Global config instance
config = AppConfig()


def validate_config() -> bool:
    """Validate required configuration"""
    errors = []
    
    if not config.openai.api_key:
        errors.append("OPENAI_API_KEY not set in .env file")
    
    if not config.milvus.is_cloud and not config.milvus.host:
        errors.append("Either MILVUS_URI/TOKEN (Zilliz Cloud) or MILVUS_HOST/PORT (local) must be set")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def print_config():
    """Print current configuration (excluding secrets)"""
    print("=" * 60)
    print("📋 Application Configuration")
    print("=" * 60)
    print(f"Title: {config.title}")
    print(f"Debug Mode: {config.debug}")
    print(f"Log Level: {config.log_level}")
    print()
    print("🤖 OpenAI:")
    print(f"  Model: {config.openai.model}")
    print(f"  Temperature: {config.openai.temperature}")
    print(f"  API Key: {'✓ Set' if config.openai.api_key else '✗ Missing'}")
    print()
    print("🗄️ Milvus:")
    if config.milvus.is_cloud:
        print(f"  Type: Zilliz Cloud (Managed)")
        print(f"  URI: {config.milvus.uri[:30]}...")
        print(f"  Token: {'✓ Set' if config.milvus.token else '✗ Missing'}")
    else:
        print(f"  Type: Self-hosted")
        print(f"  Host: {config.milvus.host}")
        print(f"  Port: {config.milvus.port}")
    print(f"  Collection: {config.milvus.collection_name}")
    print()
    print("🧠 Embeddings:")
    print(f"  Model: {config.embedding.model_name}")
    print(f"  Dimension: {config.embedding.dimension}")
    print(f"  Device: {config.embedding.device}")
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration
    print_config()
    
    if validate_config():
        print("\n✅ Configuration is valid!")
    else:
        print("\n❌ Configuration has errors. Please check your .env file.")
