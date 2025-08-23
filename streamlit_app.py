import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import time
from typing import Dict, List, Tuple, Optional, Union
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from io import BytesIO, StringIO

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                           roc_curve, auc, confusion_matrix, precision_recall_curve,
                           f1_score, precision_score, recall_score, balanced_accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from scipy.spatial.distance import cdist
from scipy import stats

# Optional imports with error handling
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    st.warning("⚠️ UMAP not available - some visualizations may be limited")

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    st.warning("⚠️ imbalanced-learn not available - using basic balancing")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Methane Leak Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .success-box {
        background-color: #D4F4DD;
        border: 1px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFC107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class Config:
    """Enhanced configuration for Streamlit deployment"""
    
    # Base paths - Streamlit compatible
    BASE_PATH = Path('./data') if Path('./data').exists() else Path('/tmp/data')
    OUTPUT_BASE = Path('./output') if not st.runtime.exists() else Path('/tmp/output')
    
    HOLE_SIZES = ['20mm', '25mm', '30mm', '40mm']
    
    # Data formats
    EXPECTED_COLUMNS = ['nodenumber', 'x-coordinate', 'y-coordinate', 'temperature']
    
    # Model parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    
    # BALANCED CLASSIFICATION PARAMETERS
    TEMPERATURE_THRESHOLD = 305
    SPATIAL_ZONES = 20
    ROLLING_WINDOW = 5
    TARGET_LEAK_RATIO = 0.4
    MIN_LEAK_RATIO = 0.2
    MAX_LEAK_RATIO = 0.6
    SENSITIVITY_LEVEL = 'balanced'
    
    @staticmethod
    def setup_directories():
        """Setup directories for Streamlit"""
        for dir_path in [Config.OUTPUT_BASE, Config.OUTPUT_BASE / 'models', 
                        Config.OUTPUT_BASE / 'plots', Config.OUTPUT_BASE / 'data']:
            dir_path.mkdir(parents=True, exist_ok=True)
        return True
    
    @staticmethod
    def handle_uploaded_files(uploaded_files):
        """Handle file uploads in Streamlit"""
        if not uploaded_files:
            return {}
        
        # Create temporary data structure
        data_structure = {}
        for hole_size in Config.HOLE_SIZES:
            data_structure[hole_size] = {
                'data_files': [],
                'file_count': 0
            }
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            # Try to determine hole size from filename
            filename = uploaded_file.name.lower()
            hole_size = None
            for hs in Config.HOLE_SIZES:
                if hs.replace('mm', '') in filename or hs in filename:
                    hole_size = hs
                    break
            
            if not hole_size:
                # Default to first hole size or let user specify
                hole_size = Config.HOLE_SIZES[0]
            
            # Save file temporarily
            temp_path = Config.BASE_PATH / hole_size / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            data_structure[hole_size]['data_files'].append(temp_path)
            data_structure[hole_size]['file_count'] += 1
        
        return data_structure

class BalancedDataLoader:
    """Enhanced data loader for Streamlit"""
    
    @staticmethod
    def parse_data_file(file_path: Path) -> pd.DataFrame:
        """Parse data file with advanced format detection"""
        try:
            # Try different parsing methods
            parsing_methods = [
                lambda: pd.read_csv(file_path, sep=',', header=None),
                lambda: pd.read_csv(file_path, sep='\s+', header=None),
                lambda: pd.read_csv(file_path, sep='\t', header=None),
                lambda: pd.read_csv(file_path, sep=None, engine='python', header=None),
            ]
            
            df = None
            for method in parsing_methods:
                try:
                    df = method()
                    if len(df.columns) >= 4 and len(df) > 0:
                        break
                except:
                    continue
            
            if df is None or df.empty:
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
            st.error(f"⚠️ Could not parse {file_path.name}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def balanced_leak_classification(df: pd.DataFrame, hole_size: str, file_idx: int = 0) -> Dict:
        """BALANCED leak classification with multiple strategies"""
        if df.empty or 'temperature' not in df.columns:
            return {'leak_status': 0, 'confidence': 0.0, 'criteria_met': [], 'method': 'empty'}
        
        temp_stats = {
            'mean': df['temperature'].mean(),
            'max': df['temperature'].max(),
            'min': df['temperature'].min(),
            'std': df['temperature'].std(),
            'q75': df['temperature'].quantile(0.75),
            'q90': df['temperature'].quantile(0.90),
            'q95': df['temperature'].quantile(0.95),
        }
        
        # Hole size factors
        hole_factors = {'20mm': 0.8, '25mm': 1.0, '30mm': 1.2, '40mm': 1.5}
        factor = hole_factors.get(hole_size, 1.0)
        
        # Multiple classification criteria
        base_threshold = Config.TEMPERATURE_THRESHOLD * factor
        criteria = {}
        criteria_met = []
        
        # Statistical outliers
        z_scores = np.abs(stats.zscore(df['temperature']))
        outlier_count = np.sum(z_scores > 2)
        outlier_ratio = outlier_count / len(df)
        criteria['outliers'] = outlier_ratio > 0.1 * factor
        if criteria['outliers']:
            criteria_met.append(f"Outlier ratio ({outlier_ratio:.1%}) significant")
        
        # High temperature ratio
        high_temp_count = np.sum(df['temperature'] > base_threshold)
        high_temp_ratio = high_temp_count / len(df)
        criteria['high_temp_ratio'] = high_temp_ratio > 0.15 * factor
        if criteria['high_temp_ratio']:
            criteria_met.append(f"High temp ratio ({high_temp_ratio:.1%}) > threshold")
        
        # Mean temperature
        mean_threshold = base_threshold - 3
        criteria['mean_temp'] = temp_stats['mean'] > mean_threshold
        if criteria['mean_temp']:
            criteria_met.append(f"Mean temp ({temp_stats['mean']:.1f}K) > {mean_threshold:.1f}K")
        
        # Temperature variability
        std_threshold = 8 * factor
        criteria['temp_variance'] = temp_stats['std'] > std_threshold
        if criteria['temp_variance']:
            criteria_met.append(f"Temp std ({temp_stats['std']:.1f}) > {std_threshold:.1f}")
        
        # Maximum temperature
        max_threshold = base_threshold + (5 * factor)
        criteria['max_temp'] = temp_stats['max'] > max_threshold
        if criteria['max_temp']:
            criteria_met.append(f"Max temp ({temp_stats['max']:.1f}K) > {max_threshold:.1f}K")
        
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
        
        # Additional boost for larger holes
        if hole_size in ['30mm', '40mm'] and criteria_count >= 2:
            is_leak = True
            confidence = min(1.0, confidence + 0.2)
        
        return {
            'leak_status': int(is_leak),
            'confidence': confidence,
            'criteria_met': criteria_met,
            'criteria_count': criteria_count,
            'method': f'balanced_{Config.SENSITIVITY_LEVEL}',
            'temp_stats': temp_stats
        }

class AdvancedFeatureEngineer:
    """Feature engineering for Streamlit app"""
    
    @staticmethod
    def create_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced thermal features"""
        df_features = df.copy()
        
        if 'temperature' not in df_features.columns:
            return df_features
        
        ambient_temp = 300.0
        df_features['temp_anomaly'] = df_features['temperature'] - ambient_temp
        df_features['temp_anomaly_abs'] = np.abs(df_features['temp_anomaly'])
        df_features['temp_normalized'] = (df_features['temperature'] - ambient_temp) / ambient_temp
        df_features['temp_log'] = np.log(df_features['temperature'])
        df_features['temp_squared'] = df_features['temperature'] ** 2
        df_features['temp_sqrt'] = np.sqrt(df_features['temperature'])
        
        # Temperature categories
        temp_bins = [0, 285, 295, 305, 315, 325, 350, np.inf]
        temp_labels = [0, 1, 2, 3, 4, 5, 6]
        df_features['temp_category'] = pd.cut(df_features['temperature'], bins=temp_bins, labels=temp_labels)
        df_features['temp_category_encoded'] = df_features['temp_category'].astype(int)
        
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
        df_processed = AdvancedFeatureEngineer.create_thermal_features(df_processed)
        df_processed = AdvancedFeatureEngineer.create_spatial_features(df_processed)
        
        # Encode categorical variables
        if 'hole_size' in df_processed.columns:
            le_hole = LabelEncoder()
            df_processed['hole_size_encoded'] = le_hole.fit_transform(df_processed['hole_size'])
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        return df_processed

class BalancedModelTrainer:
    """Model trainer for Streamlit"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.feature_names = []
        
    def prepare_balanced_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training"""
        target_col = 'leak_status'
        exclude_cols = [
            target_col, 'hole_size', 'file_name', 'nodenumber', 
            'classification_method', 'temp_category'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove low variance features
        variance_selector = VarianceThreshold(threshold=0.001)
        X_filtered = variance_selector.fit_transform(X)
        selected_features = X.columns[variance_selector.get_support()].tolist()
        
        X_final = pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
        
        self.feature_names = selected_features
        return X_final.values, y.values, selected_features
    
    def create_balanced_models(self):
        """Create model configurations"""
        return {
            'BalancedRandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=Config.RANDOM_STATE, 
                    n_jobs=-1,
                    class_weight='balanced',
                    max_depth=15
                ),
                'params': {
                    'max_depth': [10, 15, 20], 
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'BalancedGradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100, 
                    random_state=Config.RANDOM_STATE,
                    learning_rate=0.1
                ),
                'params': {
                    'learning_rate': [0.05, 0.1, 0.15], 
                    'max_depth': [3, 5, 7]
                }
            },
            'BalancedLogisticRegression': {
                'model': LogisticRegression(
                    random_state=Config.RANDOM_STATE, 
                    max_iter=2000,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10, 100]
                }
            }
        }
    
    def train_all_models(self, X_train, X_val, y_train, y_val, feature_names):
        """Train all models"""
        models_config = self.create_balanced_models()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, config) in enumerate(models_config.items()):
            status_text.text(f"Training {name}...")
            progress = (i + 1) / len(models_config)
            progress_bar.progress(progress)
            
            start_time = time.time()
            
            # Scale data for certain models
            if 'LogisticRegression' in name:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                self.scalers[name] = scaler
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                self.scalers[name] = None
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=cv, 
                scoring='balanced_accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Predictions
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val_scaled)
            y_proba = best_model.predict_proba(X_val_scaled)[:, 1] if best_model.predict_proba(X_val_scaled).shape[1] > 1 else np.full(len(y_val), 0.5)
            
            # Store results
            self.models[name] = best_model
            self.results[name] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5,
                'f1_score': f1_score(y_val, y_pred) if len(np.unique(y_val)) > 1 else 0.0,
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'training_time': training_time,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        
        # Find best model
        self.best_model = max(self.results, key=lambda x: self.results[x]['balanced_accuracy'])
        
        progress_bar.empty()
        status_text.empty()
        
        return self.results

def make_prediction(x_coord: float, y_coord: float, temperature: float, hole_size: str, trainer):
    """Make prediction using trained model"""
    if not trainer or not hasattr(trainer, 'best_model'):
        return None
    
    try:
        # Create sample dataframe
        sample_data = {
            'x-coordinate': [x_coord],
            'y-coordinate': [y_coord],
            'temperature': [temperature],
            'hole_size': [hole_size],
            'nodenumber': [1],
            'file_name': ['prediction_sample'],
            'leak_status': [0],
            'leak_confidence': [0.5],
            'classification_method': ['prediction'],
            'criteria_count': [0]
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Feature engineering
        feature_engineer = AdvancedFeatureEngineer()
        engineered_sample = feature_engineer.engineer_all_features(sample_df)
        
        # Select features
        feature_cols = [col for col in trainer.feature_names if col in engineered_sample.columns]
        X_sample = engineered_sample[feature_cols].fillna(engineered_sample[feature_cols].median())
        
        # Handle missing features
        for feature in trainer.feature_names:
            if feature not in X_sample.columns:
                X_sample[feature] = 0
        
        X_sample = X_sample[trainer.feature_names]
        
        # Scale if needed
        scaler = trainer.scalers.get(trainer.best_model)
        if scaler:
            X_sample_scaled = scaler.transform(X_sample.values)
        else:
            X_sample_scaled = X_sample.values
        
        # Make prediction
        prediction = trainer.models[trainer.best_model].predict(X_sample_scaled)[0]
        probability = trainer.models[trainer.best_model].predict_proba(X_sample_scaled)[0]
        
        leak_prob = probability[1] if len(probability) > 1 else 0.5
        confidence = max(probability) if len(probability) > 1 else 0.5
        
        return {
            'prediction': int(prediction),
            'leak_probability': leak_prob,
            'no_leak_probability': probability[0] if len(probability) > 1 else 0.5,
            'confidence': confidence,
            'risk_level': 'HIGH' if leak_prob > 0.7 else 'MEDIUM' if leak_prob > 0.4 else 'LOW',
            'model_used': trainer.best_model
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.markdown('<h1 class="main-header">🔥 Methane Leak Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Industrial Gas Leak Detection with Balanced Classification")
    
    # Initialize configuration
    Config.setup_directories()
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📁 Data Upload", "📊 Data Analysis", "🤖 Model Training", "🎯 Prediction", "📈 Results"]
    )
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = pd.DataFrame()
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📁 Data Upload":
        show_data_upload_page()
    elif page == "📊 Data Analysis":
        show_data_analysis_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "🎯 Prediction":
        show_prediction_page()
    elif page == "📈 Results":
        show_results_page()

def show_home_page():
    """Home page content"""
    st.markdown("""
    <div class="success-box">
    <h2>🎯 Welcome to the Balanced Methane Leak Detection System</h2>
    <p>This application uses advanced machine learning to detect methane leaks in industrial settings with balanced classification for optimal performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔧 Key Features
        - **Balanced Classification**: Optimized leak detection ratios
        - **Multiple ML Models**: Random Forest, Gradient Boosting, Logistic Regression
        - **Advanced Feature Engineering**: Thermal, spatial, and interaction features
        - **Real-time Predictions**: Interactive leak detection
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Data Support
        - **Multiple Hole Sizes**: 20mm, 25mm, 30mm, 40mm
        - **Flexible File Formats**: CSV, TXT, DAT
        - **Automatic Feature Detection**: X/Y coordinates, temperature
        - **Data Validation**: Automatic quality checks
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Performance
        - **Balanced Accuracy**: Primary evaluation metric
        - **Risk Categorization**: HIGH/MEDIUM/LOW risk levels
        - **Confidence Scoring**: Prediction reliability
        - **Interactive Visualization**: 2D/3D plotting
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **📁 Upload your data files** using the Data Upload page
    2. **📊 Analyze your data** to understand patterns and balance
    3. **🤖 Train machine learning models** with balanced classification
    4. **🎯 Make predictions** on new data points
    5. **📈 View comprehensive results** and performance metrics
    """)
    
    # System status
    st.markdown("---")
    st.markdown("### 📈 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Loaded", "✅ Ready" if st.session_state.data_loaded else "❌ No Data")
    with col2:
        st.metric("Model Trained", "✅ Ready" if st.session_state.model_trained else "❌ Not Trained")
    with col3:
        st.metric("Samples", len(st.session_state.combined_df) if not st.session_state.combined_df.empty else 0)
    with col4:
        leak_ratio = st.session_state.combined_df['leak_status'].mean() if 'leak_status' in st.session_state.combined_df.columns else 0
        st.metric("Leak Ratio", f"{leak_ratio:.1%}")

def show_data_upload_page():
    """Data upload page"""
    st.markdown('<h2 class="sub-header">📁 Data Upload & Processing</h2>', unsafe_allow_html=True)
    
    # File upload
    st.markdown("### 📤 Upload Data Files")
    uploaded_files = st.file_uploader(
        "Choose data files (CSV, TXT, DAT)",
        accept_multiple_files=True,
        type=['csv', 'txt', 'dat'],
        help="Upload your methane detection data files. Files should contain columns: nodenumber, x-coordinate, y-coordinate, temperature"
    )
    
    # Configuration settings
    st.markdown("### ⚙️ Configuration Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        temp_threshold = st.slider(
            "Temperature Threshold (K)",
            min_value=300,
            max_value=320,
            value=Config.TEMPERATURE_THRESHOLD,
            help="Base temperature threshold for leak detection"
        )
        
        target_ratio = st.slider(
            "Target Leak Ratio",
            min_value=0.1,
            max_value=0.6,
            value=Config.TARGET_LEAK_RATIO,
            step=0.05,
            help="Target percentage of leak samples in balanced dataset"
        )
    
    with col2:
        sensitivity_level = st.selectbox(
            "Detection Sensitivity",
            ['strict', 'balanced', 'sensitive'],
            index=1,
            help="Detection sensitivity: strict (high precision), balanced (optimal), sensitive (high recall)"
        )
        
        hole_size_filter = st.multiselect(
            "Hole Sizes to Process",
            Config.HOLE_SIZES,
            default=Config.HOLE_SIZES,
            help="Select which hole sizes to include in analysis"
        )
    
    # Update configuration
    Config.TEMPERATURE_THRESHOLD = temp_threshold
    Config.TARGET_LEAK_RATIO = target_ratio
    Config.SENSITIVITY_LEVEL = sensitivity_level
    
    if st.button("🚀 Process Data", type="primary") and uploaded_files:
        with st.spinner("Processing uploaded files..."):
            try:
                # Process uploaded files
                data_structure = Config.handle_uploaded_files(uploaded_files)
                
                # Load and process data
                data_loader = BalancedDataLoader()
                all_dataframes = []
                
                classification_summary = {'leak': 0, 'no_leak': 0}
                
                for hole_size, file_data in data_structure.items():
                    if hole_size not in hole_size_filter:
                        continue
                        
                    if file_data['file_count'] == 0:
                        continue
                    
                    st.info(f"Processing {hole_size} files...")
                    file_idx = 0
                    
                    for file_path in file_data['data_files']:
                        df = data_loader.parse_data_file(file_path)
                        if not df.empty:
                            # Balanced leak classification
                            leak_info = data_loader.balanced_leak_classification(df, hole_size, file_idx)
                            df['leak_status'] = leak_info['leak_status']
                            df['leak_confidence'] = leak_info['confidence']
                            df['classification_method'] = leak_info['method']
                            df['criteria_count'] = leak_info['criteria_count']
                            df['hole_size'] = hole_size
                            df['file_name'] = file_path.name
                            
                            all_dataframes.append(df)
                            
                            # Update summary
                            if leak_info['leak_status']:
                                classification_summary['leak'] += 1
                            else:
                                classification_summary['no_leak'] += 1
                        
                        file_idx += 1
                
                if all_dataframes:
                    combined_df = pd.concat(all_dataframes, ignore_index=True)
                    
                    # Balance dataset
                    leak_count_before = combined_df['leak_status'].sum()
                    ratio_before = leak_count_before / len(combined_df)
                    
                    # Apply balancing if needed
                    if abs(ratio_before - target_ratio) > 0.05:
                        combined_df = balance_dataset(combined_df, target_ratio)
                    
                    # Feature engineering
                    feature_engineer = AdvancedFeatureEngineer()
                    df_engineered = feature_engineer.engineer_all_features(combined_df)
                    
                    # Store in session state
                    st.session_state.combined_df = df_engineered
                    st.session_state.data_loaded = True
                    
                    # Show success message
                    st.success("✅ Data processed successfully!")
                    
                    # Show summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Samples", len(df_engineered))
                    with col2:
                        st.metric("Leak Samples", df_engineered['leak_status'].sum())
                    with col3:
                        st.metric("Features", len(df_engineered.columns))
                    with col4:
                        final_ratio = df_engineered['leak_status'].mean()
                        st.metric("Leak Ratio", f"{final_ratio:.1%}")
                    
                else:
                    st.error("❌ No valid data could be processed from uploaded files")
                    
            except Exception as e:
                st.error(f"Error processing data: {e}")

def balance_dataset(df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
    """Balance dataset to achieve target ratio"""
    if df.empty or 'leak_status' not in df.columns:
        return df
    
    current_ratio = df['leak_status'].mean()
    
    if abs(current_ratio - target_ratio) < 0.05:
        return df
    
    leak_samples = df[df['leak_status'] == 1]
    no_leak_samples = df[df['leak_status'] == 0]
    
    total_samples = len(df)
    target_leak_count = int(total_samples * target_ratio)
    target_no_leak_count = total_samples - target_leak_count
    
    if len(leak_samples) < target_leak_count:
        # Oversample leak samples
        additional_needed = target_leak_count - len(leak_samples)
        additional_leak = resample(leak_samples, 
                                 replace=True, 
                                 n_samples=additional_needed,
                                 random_state=Config.RANDOM_STATE)
        leak_samples = pd.concat([leak_samples, additional_leak], ignore_index=True)
    
    elif len(leak_samples) > target_leak_count:
        # Downsample leak samples
        leak_samples = resample(leak_samples, 
                              replace=False, 
                              n_samples=target_leak_count,
                              random_state=Config.RANDOM_STATE)
    
    if len(no_leak_samples) > target_no_leak_count:
        # Downsample no leak samples
        no_leak_samples = resample(no_leak_samples, 
                                 replace=False, 
                                 n_samples=target_no_leak_count,
                                 random_state=Config.RANDOM_STATE)
    
    balanced_df = pd.concat([leak_samples, no_leak_samples], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
    
    return balanced_df

def show_data_analysis_page():
    """Data analysis page with FIXED temperature statistics"""
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Visualization</h2>', unsafe_allow_html=True)
    
    if st.session_state.combined_df.empty:
        st.warning("⚠️ No data loaded. Please upload data first.")
        return
    
    df = st.session_state.combined_df
    
    # Dataset overview
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        leak_count = df['leak_status'].sum()
        st.metric("Leak Samples", leak_count)
    with col3:
        no_leak_count = len(df) - leak_count
        st.metric("No-Leak Samples", no_leak_count)
    with col4:
        leak_ratio = leak_count / len(df)
        st.metric("Leak Ratio", f"{leak_ratio:.1%}")
    with col5:
        st.metric("Features", len(df.columns))
    
    # Check data balance and show warning if needed
    unique_classes = df['leak_status'].unique()
    if len(unique_classes) == 1:
        if unique_classes[0] == 1:
            st.warning("⚠️ Dataset contains only LEAK samples. Consider adding no-leak samples for better balance.")
        else:
            st.warning("⚠️ Dataset contains only NO-LEAK samples. Consider adding leak samples for better balance.")
    
    # Balance quality assessment
    target_ratio = Config.TARGET_LEAK_RATIO
    actual_ratio = df['leak_status'].mean()
    deviation = abs(actual_ratio - target_ratio)
    
    if deviation < 0.05:
        balance_status = "🎉 EXCELLENT"
    elif deviation < 0.10:
        balance_status = "✅ GOOD"
    else:
        balance_status = "⚠️ NEEDS IMPROVEMENT"
    
    st.markdown(f"**Balance Quality**: {balance_status} (Target: {target_ratio:.1%}, Actual: {actual_ratio:.1%})")
    
    # Visualizations
    st.markdown("### 📊 Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperature Distribution", "📍 Spatial Distribution", "🕳️ Hole Size Analysis", "📈 Feature Correlations"])
    
    with tab1:
        # Temperature distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        leak_data = df[df['leak_status'] == 1]['temperature']
        no_leak_data = df[df['leak_status'] == 0]['temperature']
        
        # Only plot histograms if data exists for each class
        if len(no_leak_data) > 0:
            ax1.hist(no_leak_data, bins=50, alpha=0.7, label=f'No Leak ({len(no_leak_data):,})', color='blue', density=True)
        if len(leak_data) > 0:
            ax1.hist(leak_data, bins=50, alpha=0.7, label=f'Leak ({len(leak_data):,})', color='red', density=True)
        
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Density')
        ax1.set_title('Temperature Distribution by Leak Status')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by hole size
        if 'hole_size' in df.columns:
            hole_sizes = sorted(df['hole_size'].unique())
            temp_by_hole = [df[df['hole_size'] == hs]['temperature'] for hs in hole_sizes]
            
            bp = ax2.boxplot(temp_by_hole, labels=hole_sizes, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            ax2.set_xlabel('Hole Size')
            ax2.set_ylabel('Temperature (K)')
            ax2.set_title('Temperature Distribution by Hole Size')
            ax2.grid(True, alpha=0.3)
        else:
            # If no hole_size column, show overall temperature distribution
            ax2.hist(df['temperature'], bins=50, alpha=0.7, color='green')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Overall Temperature Distribution')
            ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Temperature statistics - FIXED VERSION
        st.markdown("#### 🌡️ Temperature Statistics")
        try:
            # Group by leak status and calculate statistics
            temp_grouped = df.groupby('leak_status')['temperature'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            
            # Create proper index labels based on actual groups present
            index_labels = []
            for status in temp_grouped.index:
                if status == 0:
                    index_labels.append('No Leak')
                else:
                    index_labels.append('Leak')
            
            # Only assign new index if we have the right number of labels
            if len(index_labels) == len(temp_grouped.index):
                temp_grouped.index = index_labels
            
            st.dataframe(temp_grouped, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error calculating temperature statistics: {e}")
            # Fallback: show basic statistics
            st.markdown("**Basic Temperature Statistics:**")
            st.write(f"- Mean: {df['temperature'].mean():.2f}K")
            st.write(f"- Std: {df['temperature'].std():.2f}K")
            st.write(f"- Min: {df['temperature'].min():.2f}K")
            st.write(f"- Max: {df['temperature'].max():.2f}K")
    
    with tab2:
        # Spatial distribution
        if 'x-coordinate' in df.columns and 'y-coordinate' in df.columns:
            fig = plt.figure(figsize=(12, 8))
            
            colors = ['blue' if x == 0 else 'red' for x in df['leak_status']]
            sizes = [20 if x == 1 else 5 for x in df['leak_status']]
            
            scatter = plt.scatter(df['x-coordinate'], df['y-coordinate'], 
                               c=colors, alpha=0.6, s=sizes)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Spatial Distribution (Red=Leak, Blue=No Leak)')
            plt.grid(True, alpha=0.3)
            
            # Add legend
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Leak')
            blue_patch = mpatches.Patch(color='blue', label='No Leak')
            plt.legend(handles=[red_patch, blue_patch])
            
            st.pyplot(fig)
            
            # 3D visualization with Plotly
            st.markdown("#### 🌐 3D Spatial-Temperature Distribution")
            
            # Sample data if too large
            if len(df) > 5000:
                sample_df = df.sample(n=5000, random_state=42)
                st.info("Showing sample of 5000 points for performance")
            else:
                sample_df = df
            
            colors_3d = ['blue' if x == 0 else 'red' for x in sample_df['leak_status']]
            sizes_3d = [4 if x == 1 else 2 for x in sample_df['leak_status']]
            
            # Create hover text safely
            hover_texts = []
            for i, row in sample_df.iterrows():
                status_text = '🔥 Leak' if row['leak_status'] else '✅ No Leak'
                hole_text = row.get('hole_size', 'Unknown')
                conf_text = row.get('leak_confidence', 0.0)
                hover_texts.append(f"Status: {status_text}<br>Hole: {hole_text}<br>Confidence: {conf_text:.2f}")
            
            fig_3d = go.Figure(data=go.Scatter3d(
                x=sample_df['x-coordinate'],
                y=sample_df['y-coordinate'],
                z=sample_df['temperature'],
                mode='markers',
                marker=dict(
                    color=colors_3d,
                    size=sizes_3d,
                    opacity=0.7
                ),
                text=hover_texts,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Temp: %{z:.1f}K<br>%{text}<extra></extra>'
            ))
            
            fig_3d.update_layout(
                title='3D Spatial-Temperature Distribution',
                scene=dict(
                    xaxis_title='X Coordinate',
                    yaxis_title='Y Coordinate',
                    zaxis_title='Temperature (K)'
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("⚠️ No coordinate data available for spatial visualization.")
    
    with tab3:
        # Hole size analysis
        if 'hole_size' in df.columns:
            st.markdown("#### 🕳️ Leak Distribution by Hole Size")
            
            hole_analysis = []
            for hole_size in sorted(df['hole_size'].unique()):
                subset = df[df['hole_size'] == hole_size]
                leak_count = subset['leak_status'].sum()
                total_count = len(subset)
                leak_rate = leak_count / total_count if total_count > 0 else 0
                
                hole_analysis.append({
                    'Hole Size': hole_size,
                    'Total Samples': total_count,
                    'Leak Samples': leak_count,
                    'No-Leak Samples': total_count - leak_count,
                    'Leak Rate': f"{leak_rate:.1%}",
                    'Avg Temperature': f"{subset['temperature'].mean():.1f}K"
                })
            
            hole_df = pd.DataFrame(hole_analysis)
            st.dataframe(hole_df, use_container_width=True)
            
            # Visualization
            if len(hole_df) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Stacked bar chart
                hole_sizes = hole_df['Hole Size']
                leak_counts = hole_df['Leak Samples']
                no_leak_counts = hole_df['No-Leak Samples']
                
                ax1.bar(hole_sizes, no_leak_counts, label='No Leak', color='blue', alpha=0.7)
                ax1.bar(hole_sizes, leak_counts, bottom=no_leak_counts, label='Leak', color='red', alpha=0.7)
                ax1.set_xlabel('Hole Size')
                ax1.set_ylabel('Sample Count')
                ax1.set_title('Sample Distribution by Hole Size')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Leak rate chart
                leak_rates = [float(rate.rstrip('%')) / 100 for rate in hole_df['Leak Rate']]
                bars = ax2.bar(hole_sizes, leak_rates, color=['lightcoral' if rate > 0.5 else 'lightblue' for rate in leak_rates])
                ax2.set_xlabel('Hole Size')
                ax2.set_ylabel('Leak Rate')
                ax2.set_title('Leak Rate by Hole Size')
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars, leak_rates):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{rate:.1%}', ha='center', va='bottom')
                
                st.pyplot(fig)
        else:
            st.warning("⚠️ No hole size information available.")
    
    with tab4:
        # Feature correlations
        st.markdown("#### 📈 Feature Correlations with Leak Status")
        
        if 'leak_status' in df.columns:
            numeric_features = df.select_dtypes(include=[np.number])
            correlations = []
            
            for col in numeric_features.columns:
                if col != 'leak_status' and not col.startswith('nodenumber'):
                    try:
                        corr = abs(numeric_features[col].corr(df['leak_status']))
                        if not np.isnan(corr):
                            correlations.append((col, corr))
                    except:
                        continue
            
            # Sort by correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            if correlations:
                top_features = [feat for feat, _ in correlations[:20]]
                top_corrs = [corr for _, corr in correlations[:20]]
                
                # Create correlation plot
                fig, ax = plt.subplots(figsize=(12, 8))
                colors = ['red' if corr > 0.3 else 'orange' if corr > 0.2 else 'yellow' if corr > 0.1 else 'lightblue' for corr in top_corrs]
                
                bars = ax.barh(top_features, top_corrs, color=colors)
                ax.set_xlabel('Absolute Correlation with Leak Status')
                ax.set_title('Top 20 Features by Correlation with Leak Status')
                ax.invert_yaxis()
                
                # Add correlation values
                for bar, corr in zip(bars, top_corrs):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{corr:.3f}', ha='left', va='center')
                
                st.pyplot(fig)
                
                # Show top correlations table
                st.markdown("**Top 15 Feature Correlations**")
                corr_df = pd.DataFrame(correlations[:15], columns=['Feature', 'Correlation'])
                corr_df['Correlation'] = corr_df['Correlation'].round(3)
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.warning("⚠️ No valid correlations found.")
        else:
            st.warning("⚠️ No leak status information available for correlation analysis.")

def show_model_training_page():
    """Model training page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    if st.session_state.combined_df.empty:
        st.warning("⚠️ No data loaded. Please upload data first.")
        return
    
    df = st.session_state.combined_df
    
    # Check for class balance issues
    unique_classes = df['leak_status'].nunique()
    if unique_classes < 2:
        st.error("⚠️ Cannot train models: Dataset contains only one class. Please ensure your data has both leak and no-leak samples.")
        return
    
    # Training configuration
    st.markdown("### ⚙️ Training Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        val_size = st.slider("Validation Size", 0.1, 0.3, 0.2, 0.05)
    
    with col2:
        cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 10], index=0)
        scoring_metric = st.selectbox("Primary Scoring Metric", 
                                    ["balanced_accuracy", "roc_auc", "f1", "precision", "recall"])
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    models_to_train = st.multiselect(
        "Select models to train",
        ["BalancedRandomForest", "BalancedGradientBoosting", "BalancedLogisticRegression"],
        default=["BalancedRandomForest", "BalancedGradientBoosting", "BalancedLogisticRegression"]
    )
    
    if st.button("🚀 Train Models", type="primary") and models_to_train:
        with st.spinner("Training models..."):
            try:
                # Initialize trainer
                trainer = BalancedModelTrainer()
                
                # Prepare data
                X, y, feature_names = trainer.prepare_balanced_data(df)
                
                st.info(f"Training on {len(X):,} samples with {len(feature_names)} features")
                
                # Stratified split
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=test_size + val_size, random_state=Config.RANDOM_STATE, stratify=y
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=test_size/(test_size + val_size), 
                    random_state=Config.RANDOM_STATE, stratify=y_temp
                )
                
                # Show split information
                st.markdown("#### 📊 Data Split Information")
                split_info = pd.DataFrame({
                    'Split': ['Training', 'Validation', 'Test'],
                    'Samples': [len(X_train), len(X_val), len(X_test)],
                    'Leak Ratio': [f"{y_train.mean():.1%}", f"{y_val.mean():.1%}", f"{y_test.mean():.1%}"]
                })
                st.dataframe(split_info, use_container_width=True)
                
                # Train models
                results = trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)
                
                # Store trainer in session state
                st.session_state.trainer = trainer
                st.session_state.model_trained = True
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Display results
                st.markdown("#### 📈 Training Results")
                results_data = []
                for model_name, result in results.items():
                    results_data.append({
                        'Model': model_name.replace('Balanced', ''),
                        'Accuracy': f"{result['accuracy']:.3f}",
                        'Balanced Accuracy': f"{result['balanced_accuracy']:.3f}",
                        'ROC-AUC': f"{result['roc_auc']:.3f}",
                        'F1-Score': f"{result['f1_score']:.3f}",
                        'Precision': f"{result['precision']:.3f}",
                        'Recall': f"{result['recall']:.3f}",
                        'Training Time': f"{result['training_time']:.1f}s"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Best model info
                best_model = trainer.best_model
                best_score = results[best_model]['balanced_accuracy']
                st.success(f"🏆 Best Model: {best_model.replace('Balanced', '')} (Balanced Accuracy: {best_score:.3f})")
                
                # Performance visualization
                st.markdown("#### 📊 Performance Comparison")
                
                # Create performance comparison chart
                models = list(results.keys())
                metrics = ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.ravel()
                
                for i, metric in enumerate(metrics):
                    values = [results[model][metric] for model in models]
                    model_names = [m.replace('Balanced', '') for m in models]
                    
                    bars = axes[i].bar(model_names, values, 
                                     color=['gold' if model == best_model else 'lightblue' for model in models])
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_ylabel('Score')
                    axes[i].set_ylim(0, 1)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{value:.3f}', ha='center', va='bottom')
                    
                    # Rotate x labels if needed
                    axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # ROC Curves
                st.markdown("#### 📈 ROC Curves Comparison")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                
                for i, (model_name, result) in enumerate(results.items()):
                    y_proba = result['y_proba']
                    if len(np.unique(y_val)) > 1:  # Only plot ROC if we have both classes
                        fpr, tpr, _ = roc_curve(y_val, y_proba)
                        auc_score = auc(fpr, tpr)
                        
                        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2, 
                               label=f'{model_name.replace("Balanced", "")} (AUC = {auc_score:.3f})')
                
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curves Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Feature importance (if available)
                if hasattr(trainer.models[best_model], 'feature_importances_'):
                    st.markdown(f"#### 🌟 Feature Importance - {best_model.replace('Balanced', '')}")
                    
                    importances = trainer.models[best_model].feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': trainer.feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Top 15 features
                    top_features = feature_importance.head(15)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
                    
                    bars = ax.barh(top_features['Feature'], top_features['Importance'], color=colors)
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'Top 15 Feature Importances - {best_model.replace("Balanced", "")}')
                    ax.invert_yaxis()
                    
                    # Add importance values
                    for bar, importance in zip(bars, top_features['Importance']):
                        ax.text(bar.get_width() + max(top_features['Importance'])*0.01, 
                               bar.get_y() + bar.get_height()/2, 
                               f'{importance:.4f}', ha='left', va='center')
                    
                    st.pyplot(fig)
                    
                    # Show feature importance table
                    st.dataframe(top_features, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during training: {e}")
    
    # Test set evaluation (if model is trained)
    if st.session_state.model_trained and hasattr(st.session_state, 'X_test'):
        st.markdown("---")
        st.markdown("### 🧪 Test Set Evaluation")
        
        if st.button("🔬 Evaluate on Test Set"):
            with st.spinner("Evaluating on test set..."):
                trainer = st.session_state.trainer
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Evaluate best model on test set
                best_model = trainer.models[trainer.best_model]
                scaler = trainer.scalers.get(trainer.best_model)
                
                if scaler:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_test_scaled = X_test
                
                y_pred = best_model.predict(X_test_scaled)
                y_proba = best_model.predict_proba(X_test_scaled)[:, 1] if best_model.predict_proba(X_test_scaled).shape[1] > 1 else np.full(len(y_test), 0.5)
                
                # Calculate metrics
                test_results = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
                    'f1_score': f1_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0,
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0)
                }
                
                # Display results
                st.markdown("#### 🏆 Final Test Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{test_results['accuracy']:.3f}")
                    st.metric("Balanced Accuracy", f"{test_results['balanced_accuracy']:.3f}")
                with col2:
                    st.metric("ROC-AUC", f"{test_results['roc_auc']:.3f}")
                    st.metric("F1-Score", f"{test_results['f1_score']:.3f}")
                with col3:
                    st.metric("Precision", f"{test_results['precision']:.3f}")
                    st.metric("Recall", f"{test_results['recall']:.3f}")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Raw confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['No Leak', 'Leak'], 
                           yticklabels=['No Leak', 'Leak'], ax=ax1)
                ax1.set_title('Confusion Matrix')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')
                
                # Normalized confusion matrix
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                           xticklabels=['No Leak', 'Leak'], 
                           yticklabels=['No Leak', 'Leak'], ax=ax2)
                ax2.set_title('Normalized Confusion Matrix')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Actual')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed classification report
                st.markdown("#### 📋 Classification Report")
                class_report = classification_report(y_test, y_pred, 
                                                   target_names=['No Leak', 'Leak'], 
                                                   output_dict=True)
                
                report_df = pd.DataFrame(class_report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)

def show_prediction_page():
    """Prediction page"""
    st.markdown('<h2 class="sub-header">🎯 Leak Detection Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ No model trained. Please train a model first.")
        return
    
    trainer = st.session_state.trainer
    
    st.markdown("### 🎮 Interactive Prediction Interface")
    st.markdown("Enter parameters to predict methane leak probability")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        x_coord = st.number_input("X Coordinate", value=0.0, step=0.1, 
                                help="X position coordinate")
        y_coord = st.number_input("Y Coordinate", value=0.0, step=0.1,
                                help="Y position coordinate")
    
    with col2:
        temperature = st.number_input("Temperature (K)", value=305.0, min_value=250.0, max_value=400.0,
                                    help="Temperature in Kelvin")
        hole_size = st.selectbox("Hole Size", Config.HOLE_SIZES, index=2,
                               help="Size of the hole being monitored")
    
    # Prediction button
    if st.button("🔍 Predict Leak", type="primary"):
        with st.spinner("Making prediction..."):
            result = make_prediction(x_coord, y_coord, temperature, hole_size, trainer)
            
            if result:
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                
                # Main prediction result
                if result['prediction']:
                    st.error("🔥 **LEAK DETECTED!**")
                    status_color = "red"
                else:
                    st.success("✅ **NO LEAK DETECTED**")
                    status_color = "green"
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Leak Probability", f"{result['leak_probability']:.1%}")
                with col2:
                    st.metric("No-Leak Probability", f"{result['no_leak_probability']:.1%}")
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col4:
                    risk_color = "red" if result['risk_level'] == 'HIGH' else "orange" if result['risk_level'] == 'MEDIUM' else "green"
                    st.metric("Risk Level", result['risk_level'])
                
                # Risk assessment
                st.markdown("### 🚨 Risk Assessment")
                if result['leak_probability'] > 0.8:
                    st.error("🚨 **IMMEDIATE ATTENTION REQUIRED!** High probability leak detected.")
                elif result['leak_probability'] > 0.6:
                    st.warning("⚠️ **Monitor closely** - Elevated leak probability.")
                elif result['leak_probability'] > 0.3:
                    st.info("⚠️ **Schedule inspection** when convenient.")
                else:
                    st.success("✅ **Normal operation** - No immediate concern.")
                
                # Additional information
                st.markdown("### 📊 Additional Information")
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    distance = np.sqrt(x_coord**2 + y_coord**2)
                    st.metric("Distance from Origin", f"{distance:.2f}")
                    st.metric("Model Used", result['model_used'].replace('Balanced', ''))
                
                with info_col2:
                    temp_anomaly = temperature - 300.0
                    st.metric("Temperature Anomaly", f"{temp_anomaly:.1f}K")
                    st.metric("Input Hole Size", hole_size)
                
                # Visualization
                st.markdown("### 📈 Probability Visualization")
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['leak_probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Leak Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("❌ Prediction failed. Please check your inputs and try again.")
    
    # Batch prediction
    st.markdown("---")
    st.markdown("### 📄 Batch Prediction")
    st.markdown("Upload a CSV file with multiple data points for batch prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="File should contain columns: x-coordinate, y-coordinate, temperature, hole_size"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown("#### 📊 Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🔍 Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    predictions = []
                    
                    for _, row in batch_df.iterrows():
                        result = make_prediction(
                            row.get('x-coordinate', 0),
                            row.get('y-coordinate', 0), 
                            row.get('temperature', 300),
                            row.get('hole_size', '30mm'),
                            trainer
                        )
                        
                        if result:
                            predictions.append({
                                'X-Coordinate': row.get('x-coordinate', 0),
                                'Y-Coordinate': row.get('y-coordinate', 0),
                                'Temperature': row.get('temperature', 300),
                                'Hole Size': row.get('hole_size', '30mm'),
                                'Prediction': '🔥 Leak' if result['prediction'] else '✅ No Leak',
                                'Leak Probability': f"{result['leak_probability']:.1%}",
                                'Risk Level': result['risk_level'],
                                'Confidence': f"{result['confidence']:.1%}"
                            })
                    
                    if predictions:
                        pred_df = pd.DataFrame(predictions)
                        st.markdown("#### 🎯 Batch Prediction Results")
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Summary statistics
                        leak_count = sum(1 for p in predictions if '🔥' in p['Prediction'])
                        total_count = len(predictions)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Predictions", total_count)
                        with col2:
                            st.metric("Leak Detections", leak_count)
                        with col3:
                            st.metric("No-Leak Detections", total_count - leak_count)
                        with col4:
                            st.metric("Leak Rate", f"{leak_count/total_count:.1%}")
                        
                        # Download results
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing batch file: {e}")
    
    # Prediction examples
    st.markdown("---")
    st.markdown("### 💡 Example Predictions")
    st.markdown("Try these example scenarios:")
    
    examples = [
        {"name": "Low Risk", "x": 0.0, "y": 0.0, "temp": 302.0, "hole": "20mm"},
        {"name": "Medium Risk", "x": 1.0, "y": 1.0, "temp": 310.0, "hole": "25mm"},
        {"name": "High Risk", "x": 2.0, "y": 0.5, "temp": 320.0, "hole": "40mm"},
        {"name": "Cold Spot", "x": -1.0, "y": -0.5, "temp": 295.0, "hole": "20mm"}
    ]
    
    example_cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f"🎯 {example['name']}", key=f"example_{i}"):
                st.session_state.example_x = example['x']
                st.session_state.example_y = example['y']
                st.session_state.example_temp = example['temp']
                st.session_state.example_hole = example['hole']
                st.rerun()

def show_results_page():
    """Results and model comparison page"""
    st.markdown('<h2 class="sub-header">📈 Results & Model Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ No model trained. Please train a model first.")
        return
    
    trainer = st.session_state.trainer
    df = st.session_state.combined_df
    
    # Model performance summary
    st.markdown("### 🏆 Model Performance Summary")
    
    results_data = []
    for model_name, result in trainer.results.items():
        results_data.append({
            'Model': model_name.replace('Balanced', ''),
            'Accuracy': result['accuracy'],
            'Balanced Accuracy': result['balanced_accuracy'],
            'ROC-AUC': result['roc_auc'],
            'F1-Score': result['f1_score'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'Training Time (s)': result['training_time']
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Highlight best model
    best_model_name = trainer.best_model.replace('Balanced', '')
    
    def highlight_best(row):
        if row['Model'] == best_model_name:
            return ['background-color: gold'] * len(row)
        return [''] * len(row)
    
    styled_df = results_df.style.apply(highlight_best, axis=1).format({
        'Accuracy': '{:.3f}',
        'Balanced Accuracy': '{:.3f}',
        'ROC-AUC': '{:.3f}',
        'F1-Score': '{:.3f}',
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'Training Time (s)': '{:.1f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    st.caption(f"🏆 Best Model: {best_model_name} (highlighted in gold)")
    
    # Performance radar chart
    st.markdown("### 🎯 Performance Radar Chart")
    
    best_results = trainer.results[trainer.best_model]
    metrics = ['Accuracy', 'Balanced Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
    values = [
        best_results['accuracy'],
        best_results['balanced_accuracy'], 
        best_results['roc_auc'],
        best_results['f1_score'],
        best_results['precision'],
        best_results['recall']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=metrics + [metrics[0]],
        fill='toself',
        name=best_model_name,
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title=f"Performance Radar Chart - {best_model_name}",
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data quality assessment
    st.markdown("### 📊 Data Quality Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ⚖️ Class Balance")
        leak_ratio = df['leak_status'].mean()
        target_ratio = Config.TARGET_LEAK_RATIO
        deviation = abs(leak_ratio - target_ratio)
        
        balance_quality = "Excellent" if deviation < 0.05 else "Good" if deviation < 0.1 else "Needs Improvement"
        
        st.metric("Current Leak Ratio", f"{leak_ratio:.1%}")
        st.metric("Target Ratio", f"{target_ratio:.1%}")
        st.metric("Balance Quality", balance_quality)
    
    with col2:
        st.markdown("#### 📈 Data Statistics")
        st.metric("Total Samples", len(df))
        st.metric("Features", len(df.columns))
        st.metric("Hole Sizes", len(df['hole_size'].unique()))
    
    with col3:
        st.markdown("#### 🌡️ Temperature Range")
        st.metric("Min Temperature", f"{df['temperature'].min():.1f}K")
        st.metric("Max Temperature", f"{df['temperature'].max():.1f}K")
        st.metric("Mean Temperature", f"{df['temperature'].mean():.1f}K")
    
    # Model comparison visualization
    st.markdown("### 📊 Model Comparison Visualization")
    
    # Create interactive comparison chart
    models = list(trainer.results.keys())
    model_names = [m.replace('Balanced', '') for m in models]
    
    fig = go.Figure()
    
    metrics_to_plot = ['accuracy', 'balanced_accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']
    metric_names = ['Accuracy', 'Balanced Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
    
    for i, metric in enumerate(metrics_to_plot):
        values = [trainer.results[model][metric] for model in models]
        
        fig.add_trace(go.Bar(
            name=metric_names[i],
            x=model_names,
            y=values,
            visible=(i == 1)  # Show balanced accuracy by default
        ))
    
    # Create dropdown buttons
    buttons = []
    for i, metric_name in enumerate(metric_names):
        visibility = [False] * len(metric_names)
        visibility[i] = True
        buttons.append(dict(
            label=metric_name,
            method="update",
            args=[{"visible": visibility}]
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )],
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export results
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Model Results"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📄 Download Data Summary"):
            summary_data = {
                'Total Samples': len(df),
                'Leak Samples': int(df['leak_status'].sum()),
                'No-Leak Samples': int((df['leak_status'] == 0).sum()),
                'Leak Ratio': f"{df['leak_status'].mean():.1%}",
                'Features': len(df.columns),
                'Best Model': best_model_name,
                'Best Score': f"{trainer.results[trainer.best_model]['balanced_accuracy']:.3f}"
            }
            
            summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary",
                data=csv,
                file_name="data_summary.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔍 Download Feature Importance") and hasattr(trainer.models[trainer.best_model], 'feature_importances_'):
            importances = trainer.models[trainer.best_model].feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': trainer.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            csv = feature_importance.to_csv(index=False)
            st.download_button(
                label="📥 Download Importance",
                data=csv,
                file_name="feature_importance.csv",
                mime="text/csv"
            )
    
    # System information
    st.markdown("---")
    st.markdown("### 🔧 System Information")
    
    system_info = {
        'Configuration': {
            'Temperature Threshold': f"{Config.TEMPERATURE_THRESHOLD}K",
            'Target Leak Ratio': f"{Config.TARGET_LEAK_RATIO:.1%}",
            'Sensitivity Level': Config.SENSITIVITY_LEVEL,
            'Random State': Config.RANDOM_STATE
        },
        'Libraries': {
            'UMAP Available': "✅ Yes" if UMAP_AVAILABLE else "❌ No",
            'Imbalanced-Learn Available': "✅ Yes" if IMBLEARN_AVAILABLE else "❌ No",
            'Streamlit Version': st.__version__
        }
    }
    
    for category, info in system_info.items():
        st.markdown(f"**{category}:**")
        for key, value in info.items():
            st.text(f"  {key}: {value}")

if __name__ == "__main__":
    main()