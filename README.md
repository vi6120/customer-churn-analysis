## Author

**Vikas Ramaswamy**

# Customer Churn Analysis with Python

A comprehensive machine learning project for predicting customer churn using analytics and machine learning techniques.

## Project Overview

This project analyzes customer behavior patterns to predict which customers are likely to churn (leave the service) and provides actionable business insights for customer retention strategies.

**Educational Purpose**: This project is created for educational and demonstration purposes only.

## Required Data Fields

**IMPORTANT**: Your dataset must contain these exact column names for the application to work properly:

### Mandatory Fields:
- **`Churn`** - Target variable (binary: 0/1, Yes/No, True/False)
- **`customerID`** - Unique customer identifier

### Required for Full Functionality:
- **`Contract`** - Contract type (used in business insights)
- **`tenure`** - Customer tenure in months (used in business insights)
- **`TotalCharges`** - Total charges (handled in data cleaning)

### Optional Fields:
Any additional categorical or numerical features (gender, services, charges, etc.) will be automatically processed for analysis.

## Features

- **Exploratory Data Analysis**: Comprehensive analysis of customer churn patterns
- **Machine Learning Models**: Logistic Regression, Random Forest, and XGBoost implementations
- **Interactive Dashboard**: Streamlit web application for real-time churn prediction
- **Business Insights**: Actionable recommendations based on model findings
- **Model Evaluation**: Complete performance metrics and comparison

## Tech Stack

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, XGBoost
- **Web App**: Streamlit
- **Development**: Jupyter notebooks

## Project Structure

```
customer-churn-analysis/
├── data/                          # Dataset storage
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_machine_learning_modeling.ipynb
├── models/                        # Saved ML models
├── app.py                         # Streamlit dashboard
├── utils.py                       # Utility functions
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── LICENSE                        # MIT license
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Jupyter Notebooks

Start with exploratory data analysis:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

Then proceed to machine learning modeling:
```bash
jupyter notebook notebooks/02_machine_learning_modeling.ipynb
```

### 2. Launch Streamlit Dashboard

Run the interactive web application:
```bash
streamlit run app.py
```

The dashboard provides:
- Data upload functionality
- Real-time churn prediction
- Interactive visualizations
- Business insights and recommendations

### 3. Using Sample Data

The application includes sample data generation for demonstration. Click "Use Sample Data" in the sidebar to get started immediately.

## Dataset

The project works with customer churn datasets containing:

- **Customer Demographics**: Age, gender, location
- **Service Information**: Contract type, payment method, services used
- **Usage Metrics**: Tenure, monthly charges, total charges
- **Target Variable**: Churn status (Yes/No)

### Expected Data Format

```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,InternetService,Contract,PaymentMethod,MonthlyCharges,TotalCharges,Churn
C0001,Female,0,Yes,No,1,No,DSL,Month-to-month,Electronic check,29.85,29.85,No
C0002,Male,0,No,No,34,Yes,DSL,One year,Mailed check,56.95,1889.5,No
```

## Model Performance

The project implements and compares three machine learning models:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| XGBoost | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

## Key Insights

Based on the analysis, the main churn drivers typically include:

1. **Contract Type**: Month-to-month contracts show higher churn rates
2. **Customer Tenure**: New customers (≤12 months) are at higher risk
3. **Pricing**: High monthly charges correlate with increased churn
4. **Payment Method**: Electronic check payments show higher churn rates
5. **Service Usage**: Lack of additional services increases churn probability

## Business Recommendations

1. **Retention Strategies**:
   - Target month-to-month contract customers with long-term offers
   - Implement early intervention programs for new customers
   - Develop loyalty programs for high-value customers

2. **Pricing Optimization**:
   - Review pricing strategy for high-charge customers
   - Offer personalized discounts based on churn probability

3. **Service Improvements**:
   - Encourage automatic payment methods
   - Promote additional service adoption
   - Enhance customer onboarding experience

## Deployment

### Streamlit Cloud Deployment

1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from the repository

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Contributing

This is an educational project. Feel free to fork and modify for learning purposes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**Educational Purpose Only**: This project is created for educational and demonstration purposes. The analysis and predictions should not be used for actual business decisions without proper validation and domain expertise. The dataset used may be synthetic or anonymized for learning purposes.

## Contact

For questions or suggestions regarding this educational project, please create an issue in the repository.

## Acknowledgments

- Dataset inspiration from Kaggle's Telco Customer Churn dataset
- Built with open-source libraries and tools
- Created for educational and learning purposes