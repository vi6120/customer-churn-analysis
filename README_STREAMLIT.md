# Customer Churn Analysis - Streamlit Ready

Complete customer churn analysis application in a single file, ready for Streamlit deployment.

## Quick Start

### Option 1: Single Command Execution
```bash
python run_app.py
```

### Option 2: Manual Steps
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the application
streamlit run churn_analysis_app.py
```

## Files

- `churn_analysis_app.py` - Complete application (single file)
- `run_app.py` - Single execution script
- `requirements_streamlit.txt` - Streamlined dependencies
- `README_STREAMLIT.md` - This file

## Features

✅ **Complete ML Pipeline**
- Data preprocessing and cleaning
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Model evaluation and comparison
- Feature importance analysis

✅ **Interactive Dashboard**
- Data upload or sample data generation
- Real-time model training
- Churn prediction for individual customers
- Business insights and recommendations

✅ **Streamlit Cloud Ready**
- Self-contained application
- No external dependencies
- Built-in sample data
- Educational disclaimers

## Deployment

### Streamlit Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy `churn_analysis_app.py`

### Local Development
```bash
git clone <repository>
cd customer-churn-analysis
python run_app.py
```

## Educational Purpose

This project is created for educational and demonstration purposes only. The analysis and predictions should not be used for actual business decisions without proper validation.

## License

MIT License - See LICENSE file for details.