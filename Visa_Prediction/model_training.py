import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class VisaModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n=== TRAINING LOGISTIC REGRESSION ===")
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, self.y_train)
        
        best_lr = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_lr.predict(X_test_scaled)
        y_pred_proba = best_lr.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models['Logistic Regression'] = {
            'model': best_lr,
            'scaler': self.scaler,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': grid_search.best_params_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n=== TRAINING RANDOM FOREST ===")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_rf = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_rf.predict(self.X_test)
        y_pred_proba = best_rf.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models['Random Forest'] = {
            'model': best_rf,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': grid_search.best_params_,
            'feature_importance': best_rf.feature_importances_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n=== TRAINING GRADIENT BOOSTING ===")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_gb = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_gb.predict(self.X_test)
        y_pred_proba = best_gb.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models['Gradient Boosting'] = {
            'model': best_gb,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': grid_search.best_params_,
            'feature_importance': best_gb.feature_importances_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
    def compare_models(self):
        """Compare all trained models"""
        print("\n=== MODEL COMPARISON ===")
        
        comparison_df = pd.DataFrame({
            'Model': list(self.models.keys()),
            'Accuracy': [self.models[model]['accuracy'] for model in self.models],
            'ROC AUC': [self.models[model]['roc_auc'] for model in self.models]
        })
        
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        print(comparison_df)
        
        # Select best model
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {self.best_model['accuracy']:.4f}")
        print(f"Best ROC AUC: {self.best_model['roc_auc']:.4f}")
        
        return comparison_df
        
    def plot_model_comparison(self):
        """Plot model comparison"""
        comparison_data = []
        for model_name, model_data in self.models.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': model_data['accuracy'],
                'ROC AUC': model_data['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        df_comparison.set_index('Model')['Accuracy'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # ROC AUC comparison
        df_comparison.set_index('Model')['ROC AUC'].plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Model ROC AUC Comparison')
        ax2.set_ylabel('ROC AUC')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model_data) in enumerate(self.models.items()):
            cm = confusion_matrix(self.y_test, model_data['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAccuracy: {model_data["accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model_data in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, model_data['probabilities'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {model_data["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        tree_models = ['Random Forest', 'Gradient Boosting']
        available_tree_models = [model for model in tree_models if model in self.models]
        
        if not available_tree_models:
            print("No tree-based models available for feature importance plot")
            return
        
        fig, axes = plt.subplots(1, len(available_tree_models), figsize=(8*len(available_tree_models), 6))
        
        if len(available_tree_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(available_tree_models):
            model_data = self.models[model_name]
            feature_importance = model_data['feature_importance']
            feature_names = self.X.columns
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            # Plot
            importance_df.plot(x='feature', y='importance', kind='barh', ax=axes[idx])
            axes[idx].set_title(f'{model_name} - Feature Importance')
            axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_models(self):
        """Save all trained models"""
        for model_name, model_data in self.models.items():
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model_data, filename)
            print(f"Saved {model_name} to {filename}")
        
        # Save best model separately
        joblib.dump(self.best_model, 'best_model.pkl')
        print(f"Best model ({self.best_model_name}) saved as 'best_model.pkl'")
        
    def generate_classification_reports(self):
        """Generate detailed classification reports"""
        print("\n=== DETAILED CLASSIFICATION REPORTS ===")
        
        for model_name, model_data in self.models.items():
            print(f"\n{model_name}:")
            print("="*50)
            print(classification_report(self.y_test, model_data['predictions']))
            
    def train_all_models(self):
        """Train all models and generate complete analysis"""
        print("Starting model training pipeline...")
        
        self.split_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        
        comparison_df = self.compare_models()
        
        # Generate visualizations
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_feature_importance()
        
        # Generate reports
        self.generate_classification_reports()
        
        # Save models
        self.save_models()
        
        print("\nModel training completed successfully!")
        return comparison_df

if __name__ == "__main__":
    # Load preprocessed data
    from data_preprocessing import VisaDataPreprocessor
    
    preprocessor = VisaDataPreprocessor('Visa_Predection_Dataset.csv')
    X, y, processed_df = preprocessor.preprocess()
    
    # Train models
    trainer = VisaModelTrainer(X, y)
    comparison_results = trainer.train_all_models()