"""
Detection Agent - Performs deepfake detection analysis in VoiceGuard AI
Responsible for classifying audio as real or fake using extracted features
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class DetectionAgent:
    """
    Agent responsible for deepfake detection in the multi-agent VoiceGuard AI system.
    
    This agent:
    - Receives extracted features from the Feature Agent
    - Applies multiple ML models for classification
    - Combines predictions from ensemble methods
    - Provides confidence scores and detection results
    - Supports both pre-trained and custom model training
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the Detection Agent.
        
        Args:
            model_type: Type of model to use ('random_forest', 'svm', 'gradient_boosting', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        
        # Initialize models based on type
        self._initialize_models()
        
    def _initialize_models(self):
        """
        Initialize machine learning models based on the selected model type.
        
        Each model has different strengths:
        - Random Forest: Good for feature importance, handles non-linear relationships
        - SVM: Effective in high-dimensional spaces, good for complex decision boundaries
        - Gradient Boosting: High accuracy, handles imbalanced data well
        - Ensemble: Combines multiple models for better generalization
        """
        if self.model_type == 'random_forest':
            # Random Forest - ensemble of decision trees
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,           # Number of trees in the forest
                max_depth=20,               # Maximum depth of trees
                min_samples_split=10,       # Minimum samples required to split a node
                min_samples_leaf=5,         # Minimum samples required at leaf node
                random_state=42,            # For reproducibility
                n_jobs=-1                   # Use all available cores
            )
            
        elif self.model_type == 'svm':
            # Support Vector Machine with RBF kernel
            self.models['svm'] = SVC(
                kernel='rbf',               # Radial basis function kernel
                C=1.0,                      # Regularization parameter
                gamma='scale',              # Kernel coefficient
                probability=True,           # Enable probability estimates
                random_state=42
            )
            
        elif self.model_type == 'gradient_boosting':
            # Gradient Boosting - sequential ensemble of weak learners
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=100,           # Number of boosting stages
                learning_rate=0.1,          # Step size shrinkage
                max_depth=5,                # Maximum depth of individual trees
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            
        elif self.model_type == 'ensemble':
            # Ensemble of multiple models for better performance
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['svm'] = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to a flat feature vector for ML models.
        
        This method flattens the nested feature dictionary into a 1D array
        that can be fed into machine learning models.
        
        Args:
            features: Dictionary of extracted features from Feature Agent
            
        Returns:
            1D numpy array containing all features
        """
        feature_list = []
        
        def extract_values(d, parent_key=''):
            """
            Recursively extract numeric values from nested dictionary.
            
            Args:
                d: Dictionary or list to extract values from
                parent_key: Parent key for context
            """
            if isinstance(d, dict):
                # Sort keys for consistent feature ordering
                for key in sorted(d.keys()):
                    if key == "metadata":
                        continue  # Skip metadata, not a feature
                    # Skip raw feature matrices (keep only statistics)
                    if key.endswith("_features") or key.endswith("_matrix"):
                        continue
                    extract_values(d[key], key)
            elif isinstance(d, list):
                # Only include lists that are not too long (avoid raw matrices)
                if len(d) <= 100:  # Threshold for statistical summaries
                    feature_list.extend(d)
            elif isinstance(d, (int, float)):
                # Append scalar values
                feature_list.append(d)
        
        extract_values(features)
        
        return np.array(feature_list, dtype=np.float64)
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the detection model(s) on labeled data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0 for real, 1 for fake)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training results and metrics
        """
        results = {
            "success": False,
            "metrics": {},
            "model_scores": {},
            "error": None
        }
        
        try:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42, 
                stratify=y  # Maintain class distribution
            )
            
            # Normalize features using StandardScaler
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each model
            for model_name, model in self.models.items():
                print(f"Training {model_name}...")
                
                # Train the model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions on test set
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Store results
                results["model_scores"][model_name] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "confusion_matrix": conf_matrix.tolist()
                }
                
                # Extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = model.feature_importances_
            
            # Calculate ensemble metrics if using ensemble
            if self.model_type == 'ensemble':
                ensemble_pred = self._ensemble_predict(X_test_scaled)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                ensemble_precision = precision_score(y_test, ensemble_pred, zero_division=0)
                ensemble_recall = recall_score(y_test, ensemble_pred, zero_division=0)
                ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
                
                results["model_scores"]["ensemble"] = {
                    "accuracy": float(ensemble_accuracy),
                    "precision": float(ensemble_precision),
                    "recall": float(ensemble_recall),
                    "f1_score": float(ensemble_f1)
                }
            
            # Mark as trained
            self.is_trained = True
            results["success"] = True
            results["metrics"] = {
                "num_training_samples": len(X_train),
                "num_test_samples": len(X_test),
                "num_features": X.shape[1],
                "models_trained": list(self.models.keys())
            }
            
        except Exception as e:
            results["error"] = str(e)
            results["error_type"] = type(e).__name__
        
        return results
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions by combining multiple models.
        
        Uses soft voting (average probabilities) for classification.
        This ensures the prediction matches the class with highest average probability.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted labels
        """
        # Get probabilities from all models
        proba_list = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                proba_list.append(proba)
        
        if not proba_list:
            # Fallback to hard voting if no probabilities available
            predictions = []
            for model_name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)
            predictions = np.array(predictions)
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=0, 
                arr=predictions
            )
        
        # Average probabilities across all models (soft voting)
        avg_proba = np.mean(proba_list, axis=0)
        
        # Return the class with highest average probability
        ensemble_pred = np.argmax(avg_proba, axis=1)
        
        return ensemble_pred
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict whether audio is real or fake.
        
        Main prediction method that processes features and returns
        classification results with confidence scores.
        
        Args:
            features: Dictionary of extracted features from Feature Agent
            
        Returns:
            Dictionary containing prediction results
        """
        result = {
            "success": False,
            "prediction": None,
            "confidence": None,
            "probabilities": {},
            "model_predictions": {},
            "error": None
        }
        
        try:
            # Check if model is trained
            if not self.is_trained:
                raise RuntimeError("Model has not been trained. Call train() first.")
            
            # Prepare feature vector
            feature_vector = self.prepare_feature_vector(features)
            
            # Reshape for single sample prediction
            feature_vector = feature_vector.reshape(1, -1)
            
            # Normalize features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions from all models
            model_probs = {}
            model_preds = {}
            
            for model_name, model in self.models.items():
                # Get prediction
                pred = model.predict(feature_vector_scaled)[0]
                model_preds[model_name] = int(pred)
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feature_vector_scaled)[0]
                    model_probs[model_name] = {
                        "real": float(proba[0]),
                        "fake": float(proba[1])
                    }
            
            # Ensemble prediction
            if self.model_type == 'ensemble':
                final_pred = self._ensemble_predict(feature_vector_scaled)[0]
                
                # Average probabilities across models
                avg_fake_prob = np.mean([probs["fake"] for probs in model_probs.values()])
                avg_real_prob = 1 - avg_fake_prob
                
                # Confidence should match the predicted class
                confidence = float(avg_fake_prob if final_pred == 1 else avg_real_prob)
            else:
                # Single model prediction
                final_pred = list(model_preds.values())[0]
                probs = list(model_probs.values())[0]
                # Confidence should match the predicted class
                confidence = float(probs["fake"] if final_pred == 1 else probs["real"])
            
            # Calculate probabilities for result
            if self.model_type == 'ensemble':
                avg_fake_prob = np.mean([probs["fake"] for probs in model_probs.values()])
                avg_real_prob = 1 - avg_fake_prob
                real_prob = float(avg_real_prob)
                fake_prob = float(avg_fake_prob)
            else:
                probs = list(model_probs.values())[0]
                real_prob = float(probs["real"])
                fake_prob = float(probs["fake"])
            
            # Update result
            result["success"] = True
            result["prediction"] = "fake" if final_pred == 1 else "real"
            result["confidence"] = confidence
            result["probabilities"] = {
                "real": real_prob,
                "fake": fake_prob
            }
            result["model_predictions"] = model_preds
            
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict on multiple audio samples.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        
        return results
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get feature importance from the trained model.
        
        Feature importance indicates which features are most influential
        in making classification decisions.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary containing feature importance information
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        if self.feature_importance is None:
            return {"error": "Feature importance not available for this model type"}
        
        # Create importance dictionary
        if feature_names is not None:
            importance_dict = {
                name: float(imp) 
                for name, imp in zip(feature_names, self.feature_importance)
            }
        else:
            importance_dict = {
                f"feature_{i}": float(imp) 
                for i, imp in enumerate(self.feature_importance)
            }
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return {
            "feature_importance": sorted_importance,
            "top_10_features": dict(list(sorted_importance.items())[:10])
        }
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            "models": self.models,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "feature_importance": self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.models = model_data["models"]
        self.scaler = model_data["scaler"]
        self.model_type = model_data["model_type"]
        self.is_trained = model_data["is_trained"]
        self.feature_importance = model_data.get("feature_importance")
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "models_available": list(self.models.keys()),
            "has_feature_importance": self.feature_importance is not None
        }


# Example usage
if __name__ == "__main__":
    print("Detection Agent - Example Usage")
    print("=" * 60)
    
    # Initialize the agent
    detection_agent = DetectionAgent(model_type='ensemble')
    
    print(f"\nModel Configuration:")
    print(f"  Model Type: {detection_agent.model_type}")
    print(f"  Models: {', '.join(detection_agent.models.keys())}")
    
    # Example: Create synthetic training data
    print("\n" + "-" * 60)
    print("Example: Training with synthetic data")
    print("-" * 60)
    
    # Generate synthetic features (in real scenario, these come from Feature Agent)
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    
    # Create synthetic feature vectors
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels (0 = real, 1 = fake)
    y = np.random.randint(0, 2, n_samples)
    
    print(f"\nTraining Data:")
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of features: {n_features}")
    print(f"  Real samples: {np.sum(y == 0)}")
    print(f"  Fake samples: {np.sum(y == 1)}")
    
    # Train the model
    print(f"\nTraining models...")
    training_results = detection_agent.train(X, y, test_size=0.2)
    
    if training_results["success"]:
        print(f"\n✓ Training successful!")
        print(f"\nModel Performance:")
        for model_name, scores in training_results["model_scores"].items():
            print(f"\n  {model_name.upper()}:")
            print(f"    Accuracy:  {scores['accuracy']:.4f}")
            print(f"    Precision: {scores['precision']:.4f}")
            print(f"    Recall:    {scores['recall']:.4f}")
            print(f"    F1 Score:  {scores['f1_score']:.4f}")
    
    # Example: Make predictions
    print("\n" + "-" * 60)
    print("Example: Making predictions")
    print("-" * 60)
    
    # Create synthetic test features
    test_features = {
        "mfcc": {
            "mfcc_mean": np.random.randn(13).tolist(),
            "mfcc_std": np.random.randn(13).tolist()
        },
        "spectral": {
            "spectral_centroid_mean": 2000.0,
            "spectral_flatness_mean": 0.1
        },
        "temporal": {
            "zcr_mean": 0.1,
            "rms_mean": 0.5
        },
        "prosodic": {
            "pitch_mean": 150.0,
            "jitter": 0.01,
            "shimmer": 0.05
        },
        "formant": {
            "f1": 500.0,
            "f2": 1500.0,
            "f3": 2500.0
        },
        "statistical": {
            "mean": 0.1,
            "std": 0.5,
            "skewness": 0.2
        }
    }
    
    # Make prediction
    prediction_result = detection_agent.predict(test_features)
    
    if prediction_result["success"]:
        print(f"\n✓ Prediction successful!")
        print(f"  Prediction: {prediction_result['prediction'].upper()}")
        print(f"  Confidence: {prediction_result['confidence']:.4f}")
        print(f"  Probabilities:")
        print(f"    Real: {prediction_result['probabilities']['real']:.4f}")
        print(f"    Fake: {prediction_result['probabilities']['fake']:.4f}")
        print(f"  Individual Model Predictions:")
        for model, pred in prediction_result["model_predictions"].items():
            print(f"    {model}: {'FAKE' if pred == 1 else 'REAL'}")
    
    # Get model info
    print("\n" + "-" * 60)
    print("Model Information:")
    print("-" * 60)
    model_info = detection_agent.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Detection agent example complete!")