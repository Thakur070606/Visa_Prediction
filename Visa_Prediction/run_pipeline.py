"""
Complete Visa Prediction Pipeline Runner
This script runs the entire pipeline from data preprocessing to model training.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    print("🛂 VISA PREDICTION SYSTEM - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('Visa_Predection_Dataset.csv'):
        print("❌ Error: Visa_Predection_Dataset.csv not found!")
        print("Please ensure the dataset file is in the current directory.")
        return
    
    # Step 1: Data Preprocessing
    if not run_command("python data_preprocessing.py", "Running Data Preprocessing"):
        print("❌ Data preprocessing failed. Stopping pipeline.")
        return
    
    # Step 2: EDA Analysis
    if not run_command("python eda_analysis.py", "Running Exploratory Data Analysis"):
        print("⚠️ EDA analysis failed, but continuing with model training.")
    
    # Step 3: Model Training
    if not run_command("python model_training.py", "Running Model Training"):
        print("❌ Model training failed. Stopping pipeline.")
        return
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📊 Generated Files:")
    print("- processed_visa_data.csv (cleaned dataset)")
    print("- Various EDA plots (PNG files)")
    print("- Trained models (PKL files)")
    print("- best_model.pkl (best performing model)")
    
    print("\n🚀 Next Steps:")
    print("Run the Streamlit app with: streamlit run streamlit_app.py")
    
    # Ask if user wants to launch Streamlit
    launch = input("\n🤔 Would you like to launch the Streamlit app now? (y/n): ").lower().strip()
    if launch in ['y', 'yes']:
        print("\n🌐 Launching Streamlit app...")
        try:
            subprocess.run("streamlit run streamlit_app.py", shell=True)
        except KeyboardInterrupt:
            print("\n👋 Streamlit app stopped.")

if __name__ == "__main__":
    main()