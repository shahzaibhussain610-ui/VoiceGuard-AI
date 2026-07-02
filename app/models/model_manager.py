"""
Model Manager - Handles model training, saving, loading, and inference
Provides utilities for managing machine learning models in VoiceGuard AI
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ModelManager:
    """
    Manager class for handling ML models in VoiceGuard AI.
    
    This class provides:
    - Model training and evaluation
    - Model saving and loading
    - Model versioning
    - Performance tracking
    - Batch prediction
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelManager.
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        self.feature_importance = None
        self.model_metadata = {}
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
    
    def initialize_models(self, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Initialize multiple models.
        
        Args:
            model_types: List of model types to initialize
            
        Returns:
            Dictionary of initialized models
        """
        if model_types is None:
            model_types = ['random_forest', 'svm', 'gradient_boosting']
        
        self.models = {}
        
        for model_type in model_types:
            if model_type == 'random_forest':
                self.models['rf'] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'svm':
                self.models['svm'] = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                self.models['gb'] = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
            elif model_type == 'logistic':
                self.models['lr'] = LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
        
        return self.models
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              test_size: float = 0.2,
              cross_validate: bool = True,
              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train models on provided data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0 for real, 1 for fake)
            test_size: Proportion of data for testing
            cross_validate: Whether to perform cross-validation
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        results = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "model_scores": {},
            "cross_validation": {},
            "error": None
        }
        
        try:
            # Initialize models if not already done
            if not self.models:
                self.initialize_models()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42, 
                stratify=y
            )
            
            # Normalize features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each model
            for model_name, model in self.models.items():
                print(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                results["model_scores"][model_name] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "confusion_matrix": conf_matrix.tolist()
                }
                
                # Cross-validation
                if cross_validate:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                               cv=cv_folds, scoring='f1')
                    results["cross_validation"][model_name] = {
                        "mean_f1": float(np.mean(cv_scores)),
                        "std_f1": float(np.std(cv_scores)),
                        "scores": cv_scores.tolist()
                    }
                
                # Extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = model.feature_importances_
            
            # Calculate ensemble metrics
            if len(self.models) > 1:
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
            
            # Save training history
            self.training_history.append({
                "timestamp": results["timestamp"],
                "num_samples": len(X),
                "num_features": X.shape[1],
                "scores": results["model_scores"]
            })
            
        except Exception as e:
            results["error"] = str(e)
            results["error_type"] = type(e).__name__
        
        return results
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions using majority voting.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted labels
        """
        predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        
        return ensemble_pred
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (n_samples, n_features) or single feature vector
            
        Returns:
            Prediction results dictionary
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
            if not self.is_trained:
                raise RuntimeError("Model has not been trained. Call train() first.")
            
            # Reshape for single sample if needed
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Normalize features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            model_probs = {}
            model_preds = {}
            
            for model_name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                model_preds[model_name] = int(pred)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    model_probs[model_name] = {
                        "real": float(proba[0]),
                        "fake": float(proba[1])
                    }
            
            # Ensemble prediction
            if len(self.models) > 1:
                final_pred = self._ensemble_predict(X_scaled)[0]
                avg_fake_prob = np.mean([probs["fake"] for probs in model_probs.values()])
                confidence = float(avg_fake_prob if final_pred == 1 else 1 - avg_fake_prob)
            else:
                final_pred = list(model_preds.values())[0]
                confidence = float(list(model_probs.values())[0]["fake"]
                                  if final_pred == 1
                                  else list(model_probs.values())[0]["real"])
            
            # Update result
            result["success"] = True
            result["prediction"] = "fake" if final_pred == 1 else "real"
            result["confidence"] = confidence
            result["probabilities"] = {
                "real": float(1 - avg_fake_prob) if len(self.models) > 1
                        else float(list(model_probs.values())[0]["real"]),
                "fake": float(avg_fake_prob) if len(self.models) > 1
                        else float(list(model_probs.values())[0]["fake"])
            }
            result["model_predictions"] = model_preds
            
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def save_model(self, filepath: str, include_history: bool = True):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
            include_history: Whether to include training history
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        model_data = {
            "models": self.models,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "feature_importance": self.feature_importance,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        if include_history:
            model_data["training_history"] = self.training_history
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Model information dictionary
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data["models"]
        self.scaler = model_data["scaler"]
        self.is_trained = model_data["is_trained"]
        self.feature_importance = model_data.get("feature_importance")
        self.training_history = model_data.get("training_history", [])
        
        info = {
            "loaded": True,
            "filepath": filepath,
            "timestamp": model_data.get("timestamp"),
            "version": model_data.get("version"),
            "models": list(self.models.keys()),
            "is_trained": self.is_trained
        }
        
        print(f"Model loaded from {filepath}")
        return info
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Feature importance dictionary
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
            "top_10_features": dict(list(sorted_importance.items())[:10]),
            "total_features": len(sorted_importance)
        }
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Evaluation results dictionary
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        results = {
            "success": False,
            "model_scores": {},
            "error": None
        }
        
        try:
            # Normalize features
            X_scaled = self.scaler.transform(X)
            
            # Evaluate each model
            for model_name, model in self.models.items():
                y_pred = model.predict(X_scaled)
                
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, zero_division=0)
                recall = recall_score(y, y_pred, zero_division=0)
                f1 = f1_score(y, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y, y_pred)
                
                results["model_scores"][model_name] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "confusion_matrix": conf_matrix.tolist()
                }
            
            # Ensemble evaluation
            if len(self.models) > 1:
                ensemble_pred = self._ensemble_predict(X_scaled)
                results["model_scores"]["ensemble"] = {
                    "accuracy": float(accuracy_score(y, ensemble_pred)),
                    "precision": float(precision_score(y, ensemble_pred, zero_division=0)),
                    "recall": float(recall_score(y, ensemble_pred, zero_division=0)),
                    "f1_score": float(f1_score(y, ensemble_pred, zero_division=0))
                }
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
            results["error_type"] = type(e).__name__
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current models.
        
        Returns:
            Model information dictionary
        """
        return {
            "is_trained": self.is_trained,
            "models_available": list(self.models.keys()),
            "num_models": len(self.models),
            "has_feature_importance": self.feature_importance is not None,
            "training_history_length": len(self.training_history),
            "model_dir": self.model_dir
        }
    
    def list_saved_models(self) -> List[str]:
        """
        List all saved model files.
        
        Returns:
            List of model file paths
        """
        if not os.path.exists(self.model_dir):
            return []
        
        model_files = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl') or file.endswith('.joblib'):
                model_files.append(os.path.join(self.model_dir, file))
        
        return sorted(model_files)
    
    def delete_model(self, filepath: str) -> bool:
        """
        Delete a saved model file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Model deleted: {filepath}")
                return True
            return False
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
    
    def export_model_metadata(self, filepath: str):
        """
        Export model metadata to JSON.
        
        Args:
            filepath: Path to save metadata
        """
        metadata = {
            "model_info": self.get_model_info(),
            "training_history": self.training_history,
            "exported_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model metadata exported to {filepath}")


# Convenience functions
def train_quick_model(X: np.ndarray, 
                     y: np.ndarray, 
                     model_type: str = "random_forest") -> Tuple[ModelManager, Dict[str, Any]]:
    """
    Quick training function for simple use cases.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of model to train
        
    Returns:
        Tuple of (ModelManager, training_results)
    """
    manager = ModelManager()
    manager.initialize_models([model_type])
    results = manager.train(X, y)
    
    return manager, results


def load_model_for_inference(model_path: str) -> ModelManager:
    """
    Load a model for inference.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        ModelManager with loaded model
    """
    manager = ModelManager()
    manager.load_model(model_path)
    return manager