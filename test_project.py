#!/usr/bin/env python3
"""
Test script to verify the Customer Churn Analysis project setup.
Run this script to ensure all components are working correctly.
"""

import sys
import importlib
import pandas as pd
import numpy as np

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'sklearn', 'xgboost', 'streamlit'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - FAILED")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("All packages imported successfully!")
    return True

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        import utils
        
        # Create sample data
        np.random.seed(42)
        sample_data = {
            'customerID': ['C001', 'C002', 'C003'],
            'gender': ['Male', 'Female', 'Male'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'MonthlyCharges': [50.0, 75.0, 100.0],
            'Churn': [1, 0, 1]
        }
        df = pd.DataFrame(sample_data)
        
        # Test encoding function
        df_encoded, encoders = utils.encode_categorical_features(df)
        print("✓ Categorical encoding works")
        
        # Test feature preparation
        X, y, feature_names = utils.prepare_features(df_encoded)
        print("✓ Feature preparation works")
        
        print("Utility functions working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions failed: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app can be imported."""
    print("\nTesting Streamlit app...")
    
    try:
        # Try to import the app (this will test syntax)
        import churn_analysis_app
        print("✓ Streamlit app imports successfully")
        print("To run the app: streamlit run churn_analysis_app.py")
        return True
        
    except Exception as e:
        print(f"✗ Streamlit app failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Customer Churn Analysis Project - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_utils,
        test_streamlit_app
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook notebooks/01_exploratory_data_analysis.ipynb")
        print("2. Run: streamlit run churn_analysis_app.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()