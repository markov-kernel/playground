"""Configuration module for API keys and other settings.
This module loads sensitive credentials from environment variables using python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path)

# API Keys for various LLM providers
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
JINA_API_KEY = os.getenv('JINA_API_KEY')

# Jina Reader settings
JINA_READER_ENDPOINT = os.getenv('JINA_READER_ENDPOINT')

# Databricks settings
DATABRICKS_HOST = os.getenv('DATABRICKS_HOST')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
DATABRICKS_CLUSTER_ID = os.getenv('DATABRICKS_CLUSTER_ID')