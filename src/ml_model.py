"""
ML model for price direction prediction.
Uses scikit-learn for simplicity - no deep learning dependencies.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# scikit-learn imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")

from ml_features import build_dataset, create_features, create_labels


@dataclass
class ModelResult:
    """Results from model training/evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_accuracy: float
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray


class PriceDirectionModel:
    """
    ML model for predicting price direction (up/down).
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Args:
            model_type: 'logistic', 'random_forest', or 'gradient_boosting'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

        # Initialize model
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: list = None,
              test_size: float = 0.2) -> ModelResult:
        """
        Train the model on data.

        Args:
            X: Feature matrix
            y: Labels (0=down, 1=up)
            feature_names: Names of features
            test_size: Fraction for test set

        Returns:
            ModelResult with metrics
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Split data (time-series aware - no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Handle NaN/Inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)

        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Predict
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        test_proba = self.model.predict_proba(X_test_scaled)

        # Metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)

        # Feature importance
        feature_importance = self._get_feature_importance()

        return ModelResult(
            accuracy=test_accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            train_accuracy=train_accuracy,
            feature_importance=feature_importance,
            predictions=test_pred,
            probabilities=test_proba
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict price direction."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of each class."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names))

        return dict(zip(self.feature_names, importances))

    def save(self, filepath: str):
        """Save model to file."""
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "model_type": self.model_type
            }, f)

    def load(self, filepath: str):
        """Load model from file."""
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            self.model_type = data["model_type"]
            self.is_trained = True


def train_and_evaluate(candles: dict, model_type: str = "random_forest",
                       horizon: int = 1, threshold: float = 0.1) -> Tuple[PriceDirectionModel, ModelResult]:
    """
    Train and evaluate model on candle data.

    Args:
        candles: Dict with OHLCV data
        model_type: Model type to use
        horizon: Prediction horizon
        threshold: Min % change for classification

    Returns:
        Trained model and results
    """
    # Build dataset
    X, y, feature_names = build_dataset(candles, horizon=horizon, threshold=threshold)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels: {np.sum(y==1)} up, {np.sum(y==0)} down")

    # Train model
    model = PriceDirectionModel(model_type=model_type)
    result = model.train(X, y, feature_names)

    return model, result


def compare_models(candles: dict, horizon: int = 1,
                   threshold: float = 0.1) -> Dict[str, ModelResult]:
    """
    Compare different model types.

    Returns:
        Dict mapping model name to results
    """
    X, y, feature_names = build_dataset(candles, horizon=horizon, threshold=threshold)

    results = {}
    model_types = ["logistic", "random_forest", "gradient_boosting"]

    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        model = PriceDirectionModel(model_type=model_type)
        result = model.train(X, y, feature_names)
        results[model_type] = result
        print(f"  Accuracy: {result.accuracy:.1%}")

    return results


def print_model_report(result: ModelResult, model_name: str = "Model"):
    """Print detailed model evaluation report."""
    print("\n" + "=" * 60)
    print(f"MODEL REPORT: {model_name}")
    print("=" * 60)

    print(f"\nPerformance Metrics:")
    print(f"  Train Accuracy: {result.train_accuracy:.1%}")
    print(f"  Test Accuracy:  {result.accuracy:.1%}")
    print(f"  Precision:      {result.precision:.1%}")
    print(f"  Recall:         {result.recall:.1%}")
    print(f"  F1 Score:       {result.f1:.1%}")

    # Check for overfitting
    overfit_gap = result.train_accuracy - result.accuracy
    if overfit_gap > 0.1:
        print(f"\n  Warning: Possible overfitting (train-test gap: {overfit_gap:.1%})")

    # Feature importance
    print(f"\nTop 10 Feature Importance:")
    sorted_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (name, importance) in enumerate(sorted_features[:10]):
        bar = "█" * int(importance * 50)
        print(f"  {i+1:2d}. {name:<25} {importance:.4f} {bar}")

    print("=" * 60)


if __name__ == "__main__":
    from fetch_historical import fetch_historical_candles

    print("=" * 60)
    print("ML MODEL TRAINING")
    print("=" * 60)

    # Fetch data
    print("\nFetching 90 days of hourly data...")
    candles = fetch_historical_candles(days=90, interval="1h")

    if len(candles["closes"]) == 0:
        print("No data fetched!")
        exit(1)

    print(f"Data: {len(candles['closes'])} candles")

    # Compare models
    print("\n" + "=" * 60)
    print("COMPARING MODELS")
    print("=" * 60)

    results = compare_models(candles, horizon=1, threshold=0.1)

    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)

    for name, result in sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True):
        print(f"{name:<25} {result.accuracy:>10.1%} {result.precision:>10.1%} "
              f"{result.recall:>10.1%} {result.f1:>10.1%}")

    # Best model report
    best_name = max(results.keys(), key=lambda x: results[x].accuracy)
    print_model_report(results[best_name], best_name)

    # Train best model and save
    print("\nTraining final model...")
    model, result = train_and_evaluate(candles, model_type=best_name, horizon=1, threshold=0.1)

    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "price_direction_model.pkl")
    print(f"Model saved to {model_dir / 'price_direction_model.pkl'}")
