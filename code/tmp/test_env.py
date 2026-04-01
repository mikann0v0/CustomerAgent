import os

# Create a small .env loader
def check_env():
    print(f"API key in environment before load: {os.getenv('SILICON_API_KEY') is not None}")
    # load .env
    from dotenv import load_dotenv
    # explicitly load from root dir
    root_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(root_env)
    print(f"API key in environment after load: {os.getenv('SILICON_API_KEY') is not None}")

from pathlib import Path
check_env()