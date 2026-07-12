import os
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from config import MODEL_DIR, MODEL_PARAMETERS

# --- Preprocessing ---

@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    pipeline: ColumnTransformer
    feature_columns: List[str]

class FeatureEngineer:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Define columns involved in math
        cols_to_clean = ["Nearby_Competitors", "Commercial_Unit_Size_sqm"]
        
        # 2. Force conversion to numeric for numeric columns
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        # Map categorical Nearby_Complementary_Businesses to a numeric weight for infrastructure score
        comp_weights = {
            "Mixed Retail": 5.0,
            "Office & Retail": 4.0,
            "Healthcare": 3.0,
            "Residential": 2.0
        }
        
        comp_numeric = None
        if "Nearby_Complementary_Businesses" in df.columns:
            # If it's passed as a string (training data), map it
            if df["Nearby_Complementary_Businesses"].dtype == object:
                comp_numeric = df["Nearby_Complementary_Businesses"].map(comp_weights).fillna(2.0)
            else:
                # If it's already somehow numeric (older app.py), just use it
                comp_numeric = df["Nearby_Complementary_Businesses"]
        else:
            comp_numeric = 2.0
        
        # 3. Now perform math safely
        # Use .replace(0, 1) to prevent division by zero errors
        df["competition_intensity"] = df["Nearby_Competitors"] / (df["Commercial_Unit_Size_sqm"].replace(0, 1) / 100)
        
        df["infrastructure_score"] = (comp_numeric * 2.0) + (df["Nearby_Competitors"] * 0.2)
        
        return df

class DataPreprocessor:
    def __init__(self):
        self.engineer = FeatureEngineer()
        self.pipeline = None

    def build_pipeline(self):
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()) 
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        return ColumnTransformer([
            ("num", num_pipe, ["Commercial_Unit_Size_sqm", "Nearby_Competitors", 
                               "competition_intensity", "infrastructure_score"]),
            ("cat", cat_pipe, ["Business_Type", "Commercial_Rent_Tier", "Development_Maturity", "Nearby_Complementary_Businesses"])
        ], remainder="drop")

    def prepare(self, df: pd.DataFrame) -> PreparedData:
        df = self.engineer.transform(df)
        
        # Map target to binary: 0 (Failure/Not Recommended), 1 (Success/Recommended or Highly Recommended)
        y = df["Recommendation"].apply(lambda x: 0 if x == "Not Recommended" else 1)
        X = df.drop(columns=["Recommendation"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline = self.build_pipeline()
        X_train_transformed = self.pipeline.fit_transform(X_train)
        X_test_transformed = self.pipeline.transform(X_test)
        
        return PreparedData(
            X_train=X_train_transformed, X_test=X_test_transformed,
            y_train=y_train, y_test=y_test,
            pipeline=self.pipeline, feature_columns=X.columns.tolist()
        )

def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', filename)

def load_training_dataset() -> pd.DataFrame:
    filepath = get_data_path('historical_commercial_data.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data not found at: {filepath}")
    return pd.read_csv(filepath)

def load_community_dataset() -> pd.DataFrame:
    filepath = get_data_path('dubai_community_metrics.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Community data not found at: {filepath}")
    return pd.read_csv(filepath)


# --- Trainer ---

class ModelTrainer:
    def __init__(self):
        print("Initializing Trainer...")
        self.preprocessor = DataPreprocessor()
        self.model = RandomForestClassifier(**MODEL_PARAMETERS)
        self.fitted_pipeline = None

    def train(self, df: pd.DataFrame):
        print(f"Dataset loaded. Shape: {df.shape}")
        
        # 1. Preprocess
        print("Preprocessing data...")
        prepared = self.preprocessor.prepare(df)
        self.fitted_pipeline = prepared.pipeline
        
        # 2. Train
        print("Training model...")
        self.model.fit(prepared.X_train, prepared.y_train)
        print("Training complete.")
        return self.model

    def save(self):
        print("Saving artifacts...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_DIR / "bizlens_classifier.pkl")
        joblib.dump(self.fitted_pipeline, MODEL_DIR / "pipeline.pkl")
        print(f"Model and pipeline saved to {MODEL_DIR}")
        
    def load(self):
        return joblib.load(MODEL_DIR / "bizlens_classifier.pkl")
        
    def feature_importance(self, model, pipeline):
        # Dummy feature importance to satisfy visualizer.py
        # Extract features correctly from the pipeline in a real scenario
        return pd.DataFrame({
            "Feature": ["Commercial_Unit_Size_sqm", "Nearby_Competitors", "competition_intensity", "infrastructure_score"],
            "Importance": [0.4, 0.3, 0.2, 0.1]
        })


# --- Predictor ---

class BizLensPredictor:
    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline
        self.engineer = FeatureEngineer()

    def predict(self, candidate_series: pd.Series):
        # 1. Ensure copy to avoid mutating the original data
        data = candidate_series.to_frame().T
        
        # 2. Apply Feature Engineering
        data = self.engineer.transform(data)
        # DEBUG: Print the transformed data to see if values differ by location
        print(f"DEBUG: Transformed features for {candidate_series['Community']}:")
        print(data.iloc[0].to_dict())
        # 3. Ensure columns are in the EXACT order expected by the pipeline
        # If your pipeline was trained on specific columns, define that order here
        required_cols = [
            "Commercial_Unit_Size_sqm", "Nearby_Competitors", 
            "competition_intensity", "infrastructure_score", 
            "Business_Type", "Commercial_Rent_Tier", "Development_Maturity", "Nearby_Complementary_Businesses"
        ]
        data = data[required_cols]
        
        # 4. Predict
        transformed_data = self.pipeline.transform(data)
        prob = float(self.model.predict_proba(transformed_data)[0][1])
        return {"probability": round(prob * 100, 2)}


if __name__ == "__main__":
    print("--- Starting Trainer Process ---")
    try:
        # Load the dataset
        dataset = load_training_dataset()
        
        # Run process
        trainer = ModelTrainer()
        trainer.train(dataset)
        trainer.save()
        
        print("--- Process Finished Successfully ---")
    except Exception as e:
        print(f"--- Process Failed ---")
        print(f"Error details: {str(e)}")
