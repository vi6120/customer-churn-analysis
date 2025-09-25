#!/usr/bin/env python3
"""
Single execution script for Customer Churn Analysis.
This script sets up and runs the complete application in order.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install the packages we need."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        return False

def run_streamlit_app():
    """Start the web app."""
    print("Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "churn_analysis_app.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")

def main():
    """Run the complete setup and launch."""
    print("Customer Churn Analysis - Complete Application")
    print("=" * 50)
    
    # Make sure we're in the right place
    if not os.path.exists("churn_analysis_app.py"):
        print("Error: churn_analysis_app.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Install what we need
    if not install_requirements():
        print("Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Setup complete! Starting the application...")
    print("The app will open in your default web browser")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Start the app
    run_streamlit_app()

if __name__ == "__main__":
    main()

