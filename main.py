from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import pickle
import joblib
from pathlib import Path
import logging
from datetime import datetime
import asyncio
import uvicorn
from io import StringIO

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="🔥 Methane Leak Detection API",
    description="AI-Powered Industrial Gas Leak Detection with Balanced Classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    BASE_PATH = Path('./data')
    MODEL_PATH = Path('./models')
    HOLE_SIZES = ['20mm', '25mm', '30mm', '40mm']
    EXPECTED_COLUMNS = ['nodenumber', 'x-coordinate', 'y-coordinate', 'temperature']
    TEMPERATURE_THRESHOLD = 305
    TARGET_LEAK_RATIO = 0.4
    SENSITIVITY_LEVEL = 'balanced'
    RANDOM_STATE = 42
    
    @classmethod
    def setup_directories(cls):
        cls.BASE_PATH.mkdir(exist_ok=True)
        cls.MODEL_PATH.mkdir(exist_ok=True)

# Pydantic models for API
class PredictionRequest(BaseModel):
    x_coordinate: float
    y_coordinate: float
    temperature: float
    hole_size: str = "30mm"

class BatchPredictionRequest(BaseModel):
    data_points: List[PredictionRequest]

class PredictionResponse(BaseModel):
    prediction: int
    leak_probability: float
    no_leak_probability: float
    confidence: float
    risk_level: str
    model_used: str

class TrainingRequest(BaseModel):
    target_leak_ratio: Optional[float] = 0.4
    sensitivity_level: Optional[str] = 'balanced'
    test_size: Optional[float] = 0.2
    models_to_train: Optional[List[str]] = ['RandomForest', 'GradientBoosting', 'LogisticRegression']

class TrainingResponse(BaseModel):
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None
    best_model: Optional[str] = None

# Global variables for model storage
trained_models = {}
model_scalers = {}
feature_names = []
best_model_name = None
training_data = pd.DataFrame()

class BalancedDataProcessor:
    """Data processing for balanced classification"""
    
    @staticmethod
    def parse_data_file(file_content: str) -> pd.DataFrame:
        """Parse uploaded data file content"""
        try:
            # Try different separators
            for separator in [',', '\t', '\s+', ';']:
                try:
                    if separator == '\s+':
                        df = pd.read_csv(StringIO(file_content), sep=separator, engine='python', header=None)
                    else:
                        df = pd.read_csv(StringIO(file_content), sep=separator, header=None)
                    
                    if len(df.columns) >= 4 and len(df) > 0:
                        break
                except:
                    continue
            else:
                return pd.DataFrame()
            
            # Assign column names
            if len(df.columns) >= 4:
                df.columns = Config.EXPECTED_COLUMNS[:len(df.columns)]
                if len(df.columns) > 4:
                    additional_cols = [f'feature_{i}' for i in range(4, len(df.columns))]
                    df.columns = Config.EXPECTED_COLUMNS + additional_cols
            
            # Data cleaning
            df = df.dropna()
            
            # Ensure numeric columns
            numeric_columns = ['x-coordinate', 'y-coordinate', 'temperature']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # Validate temperature values
            if 'temperature' in df.columns:
                df = df[(df['temperature'] > 250) & (df['temperature'] < 1000)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def balanced_leak_classification(df: pd.DataFrame, hole_size: str, file_idx: int = 0) -> Dict:
        """Balanced leak classification with multiple strategies"""
        if df.empty or 'temperature' not in df.columns:
            return {'leak_status': 0, 'confidence': 0.0, 'method': 'empty'}
        
        temp_stats = {
            'mean': df['temperature'].mean(),
            'max': df['temperature'].max(),
            'std': df['temperature'].std(),
        }
        
        # Hole size factors
        hole_factors = {'20mm': 0.8, '25mm': 1.0, '30mm': 1.2, '40mm': 1.5}
        factor = hole_factors.get(hole_size, 1.0)
        
        # Multiple classification criteria
        base_threshold = Config.TEMPERATURE_THRESHOLD * factor
        criteria = {}
        
        # Statistical outliers
        z_scores = np.abs(stats.zscore(df['temperature']))
        outlier_ratio = np.sum(z_scores > 2) / len(df)
        criteria['outliers'] = outlier_ratio > 0.1 * factor
        
        # High temperature ratio
        high_temp_ratio = np.sum(df['temperature'] > base_threshold) / len(df)
        criteria['high_temp_ratio'] = high_temp_ratio > 0.15 * factor
        
        # Mean temperature
        criteria['mean_temp'] = temp_stats['mean'] > (base_threshold - 3)
        
        # Temperature variability
        criteria['temp_variance'] = temp_stats['std'] > (8 * factor)
        
        # Decision logic
        criteria_count = sum(criteria.values())
        total_criteria = len(criteria)
        confidence = criteria_count / total_criteria
        
        if Config.SENSITIVITY_LEVEL == 'strict':
            is_leak = criteria_count >= (total_criteria - 1)
        elif Config.SENSITIVITY_LEVEL == 'sensitive':
            is_leak = criteria_count >= 1
        else:  # 'balanced'
            is_leak = criteria_count >= (total_criteria // 2)
        
        return {
            'leak_status': int(is_leak),
            'confidence': confidence,
            'method': f'balanced_{Config.SENSITIVITY_LEVEL}'
        }

class FeatureEngineer:
    """Feature engineering for leak detection"""
    
    @staticmethod
    def create_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create thermal features"""
        df_features = df.copy()
        
        if 'temperature' not in df_features.columns:
            return df_features
        
        ambient_temp = 300.0
        df_features['temp_anomaly'] = df_features['temperature'] - ambient_temp
        df_features['temp_anomaly_abs'] = np.abs(df_features['temp_anomaly'])
        df_features['temp_normalized'] = (df_features['temperature'] - ambient_temp) / ambient_temp
        df_features['temp_log'] = np.log(df_features['temperature'])
        df_features['temp_squared'] = df_features['temperature'] ** 2
        
        # Binary indicators
        df_features['is_above_ambient'] = (df_features['temperature'] > ambient_temp + 5).astype(int)
        df_features['is_hot_spot'] = (df_features['temperature'] > ambient_temp + 15).astype(int)
        df_features['is_very_hot'] = (df_features['temperature'] > ambient_temp + 25).astype(int)
        
        return df_features
    
    @staticmethod
    def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features"""
        if 'x-coordinate' not in df.columns or 'y-coordinate' not in df.columns:
            return df
        
        # Distance features
        df['distance_from_origin'] = np.sqrt(df['x-coordinate']**2 + df['y-coordinate']**2)
        df['manhattan_distance'] = np.abs(df['x-coordinate']) + np.abs(df['y-coordinate'])
        
        # Polar coordinates
        df['angle_rad'] = np.arctan2(df['y-coordinate'], df['x-coordinate'])
        df['angle_deg'] = np.degrees(df['angle_rad'])
        
        # Quadrants
        df['quadrant'] = ((df['x-coordinate'] >= 0).astype(int) * 2 + 
                         (df['y-coordinate'] >= 0).astype(int))
        
        return df
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering"""
        if df.empty:
            return df
        
        df_processed = df.copy()
        df_processed = FeatureEngineer.create_thermal_features(df_processed)
        df_processed = FeatureEngineer.create_spatial_features(df_processed)
        
        # Encode categorical variables
        if 'hole_size' in df_processed.columns:
            le_hole = LabelEncoder()
            df_processed['hole_size_encoded'] = le_hole.fit_transform(df_processed['hole_size'])
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        return df_processed

class ModelTrainer:
    """Model training for leak detection"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
    
    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for training"""
        target_col = 'leak_status'
        exclude_cols = [target_col, 'hole_size', 'file_name', 'nodenumber']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X.values, y.values, numeric_cols
    
    def create_model_configs(self):
        """Create model configurations"""
        return {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=Config.RANDOM_STATE,
                    class_weight='balanced'
                ),
                'params': {
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=Config.RANDOM_STATE
                ),
                'params': {
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=Config.RANDOM_STATE,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10, 100]
                }
            }
        }
    
    def train_models(self, X_train, X_val, y_train, y_val, feature_names, models_to_train):
        """Train selected models"""
        model_configs = self.create_model_configs()
        
        for model_name in models_to_train:
            if model_name not in model_configs:
                continue
                
            config = model_configs[model_name]
            
            # Scale data for LogisticRegression
            if model_name == 'LogisticRegression':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                self.scalers[model_name] = None
            
            # Grid search
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='balanced_accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Predictions
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val_scaled)
            y_proba = best_model.predict_proba(X_val_scaled)[:, 1]
            
            # Store results
            self.models[model_name] = best_model
            self.results[model_name] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
                'f1_score': f1_score(y_val, y_pred),
                'best_params': grid_search.best_params_
            }
        
        # Find best model
        best_model = max(self.results, key=lambda x: self.results[x]['balanced_accuracy'])
        return best_model, self.results

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🔥 Methane Leak Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "train": "/train",
            "upload_data": "/upload",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global best_model_name
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": best_model_name is not None,
        "data_loaded": not training_data.empty
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_leak(request: PredictionRequest):
    """Single prediction endpoint"""
    global trained_models, model_scalers, feature_names, best_model_name
    
    if best_model_name is None or best_model_name not in trained_models:
        raise HTTPException(status_code=400, detail="No trained model available. Please train a model first.")
    
    try:
        # Create sample dataframe
        sample_data = {
            'x-coordinate': [request.x_coordinate],
            'y-coordinate': [request.y_coordinate],
            'temperature': [request.temperature],
            'hole_size': [request.hole_size],
            'nodenumber': [1],
            'file_name': ['api_prediction'],
            'leak_status': [0]  # Dummy value
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Feature engineering
        engineered_sample = FeatureEngineer.engineer_all_features(sample_df)
        
        # Select features
        available_features = [col for col in feature_names if col in engineered_sample.columns]
        X_sample = engineered_sample[available_features]
        
        # Handle missing features
        for feature in feature_names:
            if feature not in X_sample.columns:
                X_sample[feature] = 0
        
        X_sample = X_sample[feature_names].fillna(0)
        
        # Scale if needed
        scaler = model_scalers.get(best_model_name)
        if scaler:
            X_sample_scaled = scaler.transform(X_sample.values)
        else:
            X_sample_scaled = X_sample.values
        
        # Make prediction
        model = trained_models[best_model_name]
        prediction = model.predict(X_sample_scaled)[0]
        probabilities = model.predict_proba(X_sample_scaled)[0]
        
        leak_prob = probabilities[1]
        no_leak_prob = probabilities[0]
        confidence = max(probabilities)
        
        # Determine risk level
        if leak_prob > 0.7:
            risk_level = 'HIGH'
        elif leak_prob > 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return PredictionResponse(
            prediction=int(prediction),
            leak_probability=float(leak_prob),
            no_leak_probability=float(no_leak_prob),
            confidence=float(confidence),
            risk_level=risk_level,
            model_used=best_model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    global trained_models, model_scalers, feature_names, best_model_name
    
    if best_model_name is None or best_model_name not in trained_models:
        raise HTTPException(status_code=400, detail="No trained model available. Please train a model first.")
    
    try:
        predictions = []
        
        for data_point in request.data_points:
            # Use the single prediction logic
            pred_request = PredictionRequest(**data_point.dict())
            result = await predict_leak(pred_request)
            predictions.append(result.dict())
        
        # Calculate summary statistics
        leak_count = sum(1 for p in predictions if p['prediction'] == 1)
        total_count = len(predictions)
        
        return {
            "predictions": predictions,
            "summary": {
                "total_predictions": total_count,
                "leak_detections": leak_count,
                "no_leak_detections": total_count - leak_count,
                "leak_rate": leak_count / total_count if total_count > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/upload")
async def upload_data(files: List[UploadFile] = File(...)):
    """Upload and process data files"""
    global training_data
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        all_dataframes = []
        processor = BalancedDataProcessor()
        
        for file in files:
            if not file.filename.endswith(('.csv', '.txt', '.dat')):
                continue
                
            content = await file.read()
            file_content = content.decode('utf-8')
            
            # Parse file
            df = processor.parse_data_file(file_content)
            
            if not df.empty:
                # Determine hole size from filename
                hole_size = '30mm'  # default
                for hs in Config.HOLE_SIZES:
                    if hs.replace('mm', '') in file.filename.lower() or hs in file.filename.lower():
                        hole_size = hs
                        break
                
                # Classify leaks
                leak_info = processor.balanced_leak_classification(df, hole_size)
                df['leak_status'] = leak_info['leak_status']
                df['leak_confidence'] = leak_info['confidence']
                df['hole_size'] = hole_size
                df['file_name'] = file.filename
                
                all_dataframes.append(df)
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Feature engineering
            engineered_df = FeatureEngineer.engineer_all_features(combined_df)
            
            # Store globally
            training_data = engineered_df
            
            # Statistics
            leak_ratio = engineered_df['leak_status'].mean()
            
            return {
                "status": "success",
                "message": "Data uploaded and processed successfully",
                "statistics": {
                    "total_samples": len(engineered_df),
                    "leak_samples": int(engineered_df['leak_status'].sum()),
                    "no_leak_samples": int((engineered_df['leak_status'] == 0).sum()),
                    "leak_ratio": float(leak_ratio),
                    "features": len(engineered_df.columns),
                    "files_processed": len(all_dataframes)
                }
            }
        else:
            raise HTTPException(status_code=400, detail="No valid data found in uploaded files")
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_models(background_tasks: BackgroundTasks, request: TrainingRequest):
    """Train machine learning models"""
    global training_data, trained_models, model_scalers, feature_names, best_model_name
    
    if training_data.empty:
        raise HTTPException(status_code=400, detail="No training data available. Please upload data first.")
    
    try:
        # Update configuration
        Config.TARGET_LEAK_RATIO = request.target_leak_ratio
        Config.SENSITIVITY_LEVEL = request.sensitivity_level
        
        # Prepare data
        trainer = ModelTrainer()
        X, y, feature_names_list = trainer.prepare_data(training_data)
        
        # Store feature names globally
        feature_names = feature_names_list
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=request.test_size, 
            random_state=Config.RANDOM_STATE, 
            stratify=y
        )
        
        # Train models
        best_model, results = trainer.train_models(
            X_train, X_val, y_train, y_val, 
            feature_names_list, request.models_to_train
        )
        
        # Store models globally
        trained_models = trainer.models
        model_scalers = trainer.scalers
        best_model_name = best_model
        
        # Save models to disk
        model_info = {}
        for model_name, model in trained_models.items():
            model_path = Config.MODEL_PATH / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            
            if model_name in model_scalers and model_scalers[model_name]:
                scaler_path = Config.MODEL_PATH / f"{model_name}_scaler.pkl"
                joblib.dump(model_scalers[model_name], scaler_path)
            
            model_info[model_name] = results[model_name]
        
        # Save feature names and best model info
        with open(Config.MODEL_PATH / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
        
        with open(Config.MODEL_PATH / "best_model.json", "w") as f:
            json.dump({"best_model": best_model_name}, f)
        
        return TrainingResponse(
            status="success",
            message=f"Models trained successfully. Best model: {best_model_name}",
            results=model_info,
            best_model=best_model_name
        )
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about trained models"""
    global trained_models, best_model_name, training_data, feature_names
    
    if not trained_models:
        return {
            "status": "no_models",
            "message": "No models trained yet",
            "models_available": [],
            "best_model": None
        }
    
    model_info = {}
    for model_name, model in trained_models.items():
        model_info[model_name] = {
            "type": type(model).__name__,
            "is_best": model_name == best_model_name,
            "has_scaler": model_name in model_scalers and model_scalers[model_name] is not None
        }
    
    return {
        "status": "models_available",
        "models": model_info,
        "best_model": best_model_name,
        "feature_count": len(feature_names),
        "training_samples": len(training_data) if not training_data.empty else 0,
        "data_loaded": not training_data.empty
    }

@app.post("/model/load")
async def load_saved_models():
    """Load models from disk"""
    global trained_models, model_scalers, feature_names, best_model_name
    
    try:
        if not Config.MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="No saved models found")
        
        # Load feature names
        feature_file = Config.MODEL_PATH / "feature_names.json"
        if feature_file.exists():
            with open(feature_file, "r") as f:
                feature_names = json.load(f)
        
        # Load best model info
        best_model_file = Config.MODEL_PATH / "best_model.json"
        if best_model_file.exists():
            with open(best_model_file, "r") as f:
                best_model_info = json.load(f)
                best_model_name = best_model_info["best_model"]
        
        # Load models
        loaded_models = []
        for model_file in Config.MODEL_PATH.glob("*.pkl"):
            if "scaler" not in model_file.name:
                model_name = model_file.stem
                model = joblib.load(model_file)
                trained_models[model_name] = model
                loaded_models.append(model_name)
                
                # Load scaler if exists
                scaler_file = Config.MODEL_PATH / f"{model_name}_scaler.pkl"
                if scaler_file.exists():
                    model_scalers[model_name] = joblib.load(scaler_file)
                else:
                    model_scalers[model_name] = None
        
        if not loaded_models:
            raise HTTPException(status_code=404, detail="No valid model files found")
        
        return {
            "status": "success",
            "message": f"Loaded {len(loaded_models)} models",
            "models_loaded": loaded_models,
            "best_model": best_model_name,
            "feature_count": len(feature_names)
        }
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.delete("/model/clear")
async def clear_models():
    """Clear all loaded models"""
    global trained_models, model_scalers, feature_names, best_model_name, training_data
    
    trained_models.clear()
    model_scalers.clear()
    feature_names.clear()
    best_model_name = None
    training_data = pd.DataFrame()
    
    return {
        "status": "success",
        "message": "All models and data cleared"
    }

@app.get("/data/stats")
async def get_data_stats():
    """Get statistics about loaded data"""
    global training_data
    
    if training_data.empty:
        return {
            "status": "no_data",
            "message": "No data loaded"
        }
    
    stats = {
        "total_samples": len(training_data),
        "features": len(training_data.columns),
        "leak_samples": int(training_data['leak_status'].sum()) if 'leak_status' in training_data.columns else 0,
        "no_leak_samples": int((training_data['leak_status'] == 0).sum()) if 'leak_status' in training_data.columns else 0,
        "leak_ratio": float(training_data['leak_status'].mean()) if 'leak_status' in training_data.columns else 0,
        "temperature_stats": {
            "min": float(training_data['temperature'].min()),
            "max": float(training_data['temperature'].max()),
            "mean": float(training_data['temperature'].mean()),
            "std": float(training_data['temperature'].std())
        } if 'temperature' in training_data.columns else None,
        "hole_sizes": training_data['hole_size'].unique().tolist() if 'hole_size' in training_data.columns else []
    }
    
    return {
        "status": "data_available",
        "statistics": stats
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    Config.setup_directories()
    logger.info("🔥 Methane Leak Detection API started")
    logger.info("Available endpoints:")
    logger.info("  - GET  /health - Health check")
    logger.info("  - POST /upload - Upload data files")
    logger.info("  - POST /train - Train models")
    logger.info("  - POST /predict - Single prediction")
    logger.info("  - POST /predict/batch - Batch prediction")
    logger.info("  - GET  /model/info - Model information")
    logger.info("  - POST /model/load - Load saved models")
    logger.info("  - GET  /docs - API documentation")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )