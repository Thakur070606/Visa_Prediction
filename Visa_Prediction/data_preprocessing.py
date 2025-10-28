import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class VisaDataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset"""
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded with shape: {self.df.shape}")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n=== BASIC DATASET INFO ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nTarget distribution:\n{self.df['case_status'].value_counts()}")
        
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n=== HANDLING MISSING VALUES ===")
        missing_before = self.df.isnull().sum().sum()
        
        # Fill missing values based on column type
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")
        print(f"Missing values after: {missing_after}")
        
    def handle_outliers(self):
        """Handle outliers in numerical columns"""
        print("\n=== HANDLING OUTLIERS ===")
        numerical_cols = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            # Cap outliers instead of removing them
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
            
            print(f"{col}: {outliers_before} outliers capped")
    
    def feature_engineering(self):
        """Create new features"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Company age
        current_year = 2024
        self.df['company_age'] = current_year - self.df['yr_of_estab']
        
        # Wage per hour (normalize all wages to hourly)
        self.df['wage_per_hour'] = np.where(
            self.df['unit_of_wage'] == 'Year',
            self.df['prevailing_wage'] / (52 * 40),  # Assuming 40 hours/week
            self.df['prevailing_wage']
        )
        
        # Employee size category
        self.df['company_size'] = pd.cut(
            self.df['no_of_employees'],
            bins=[0, 50, 500, 5000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Enterprise']
        )
        
        print("New features created: company_age, wage_per_hour, company_size")
        
    def encode_categorical_variables(self):
        """Encode categorical variables"""
        print("\n=== ENCODING CATEGORICAL VARIABLES ===")
        
        categorical_cols = ['continent', 'education_of_employee', 'has_job_experience', 
                          'requires_job_training', 'region_of_employment', 'unit_of_wage', 
                          'full_time_position', 'company_size']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}")
        
        # Encode target variable
        le_target = LabelEncoder()
        self.df['case_status_encoded'] = le_target.fit_transform(self.df['case_status'])
        self.label_encoders['case_status'] = le_target
        
    def prepare_features(self):
        """Prepare final feature set for modeling"""
        print("\n=== PREPARING FEATURES ===")
        
        # Select features for modeling
        feature_cols = [
            'continent_encoded', 'education_of_employee_encoded', 'has_job_experience_encoded',
            'requires_job_training_encoded', 'no_of_employees', 'company_age',
            'region_of_employment_encoded', 'wage_per_hour', 'full_time_position_encoded',
            'company_size_encoded'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        X = self.df[available_features]
        y = self.df['case_status_encoded']
        
        # Scale numerical features
        numerical_features = ['no_of_employees', 'company_age', 'wage_per_hour']
        numerical_features = [col for col in numerical_features if col in X.columns]
        
        if numerical_features:
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        print(f"Final feature set shape: {X.shape}")
        print(f"Features used: {list(X.columns)}")
        
        return X, y
    
    def preprocess(self):
        """Run complete preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        self.load_data()
        self.basic_info()
        self.handle_missing_values()
        self.handle_outliers()
        self.feature_engineering()
        self.encode_categorical_variables()
        X, y = self.prepare_features()
        
        print("\nPreprocessing completed successfully!")
        return X, y, self.df

if __name__ == "__main__":
    preprocessor = VisaDataPreprocessor('Visa_Predection_Dataset.csv')
    X, y, processed_df = preprocessor.preprocess()
    
    # Save processed data
    processed_df.to_csv('processed_visa_data.csv', index=False)
    print("\nProcessed data saved to 'processed_visa_data.csv'")