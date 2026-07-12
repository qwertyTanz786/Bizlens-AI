from pathlib import Path

# Define directories
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "bizlens_classifier.pkl"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl" # Add this for your pipeline

# Model Hyperparameters
MODEL_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

# AI/API Constants
AI_API_URL = "http://localhost:11434/api/generate"
AI_MODEL_NAME = "llama3"

# Other constants
RANDOM_STATE = 42
TEST_SIZE = 0.2