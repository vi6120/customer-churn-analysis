"""
Streamlit app for Customer Churn Analysis.
Interactive dashboard for analyzing customer churn patterns and predicting churn probability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import utils

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Customer Churn Analysis Dashboard")
st.markdown("""
This dashboard analyzes customer churn patterns and predicts which customers are likely to leave.
Upload your customer data to get started with the analysis.

**Educational Purpose Only**: This tool is for learning and demonstration purposes.
""")

# Sidebar for file upload and model selection
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with customer data including churn labels"
)

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Logistic Regression"],
    help="Choose the machine learning model for churn prediction"
)

# Sample data option
if st.sidebar.button("Use Sample Data"):
    st.sidebar.info("Using Telco Customer Churn sample data")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'customerID': [f'C{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples),
    }
    
    # Create churn based on some logic
    churn_prob = (
        (sample_data['Contract'] == 'Month-to-month') * 0.3 +
        (sample_data['tenure'] < 12) * 0.2 +
        (sample_data['MonthlyCharges'] > 80) * 0.2 +
        np.random.random(n_samples) * 0.3
    )
    sample_data['Churn'] = (churn_prob > 0.5).astype(int)
    
    uploaded_file = pd.DataFrame(sample_data)

# Main content
if uploaded_file is not None:
    # Load data
    if isinstance(uploaded_file, pd.DataFrame):
        df = uploaded_file.copy()
    else:
        df = pd.read_csv(uploaded_file)
    
    # Clean data
    df = utils.load_and_clean_data(df) if not isinstance(uploaded_file, pd.DataFrame) else df
    
    # Display basic info
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    with col3:
        st.metric("Features", len(df.columns) - 1)
    with col4:
        st.metric("Churned Customers", df['Churn'].sum())
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Exploratory Data Analysis
    st.header("Exploratory Data Analysis")
    
    # Churn distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_churn = utils.plot_churn_distribution(df)
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with col2:
        # Churn by categorical feature
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        if 'customerID' in categorical_features:
            categorical_features.remove('customerID')
        
        if categorical_features:
            selected_feature = st.selectbox("Analyze Churn by Feature", categorical_features)
            fig_feature = utils.plot_churn_by_feature(df, selected_feature)
            st.plotly_chart(fig_feature, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        fig_corr = px.imshow(
            numeric_df.corr(),
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Machine Learning Model
    st.header("Churn Prediction Model")
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            df_encoded, encoders = utils.encode_categorical_features(df)
            X, y, feature_names = utils.prepare_features(df_encoded)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features for Logistic Regression
            if model_type == "Logistic Regression":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = LogisticRegression(random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Store scaler for later use
                st.session_state['scaler'] = scaler
                X_test_for_eval = X_test_scaled
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                X_test_for_eval = X_test
            
            # Evaluate model
            metrics = utils.evaluate_model(model, X_test_for_eval, y_test)
            
            # Store model and data in session state
            st.session_state['model'] = model
            st.session_state['feature_names'] = feature_names
            st.session_state['encoders'] = encoders
            st.session_state['metrics'] = metrics
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['model_type'] = model_type
        
        st.success("Model trained successfully!")
    
    # Display model results if available
    if 'model' in st.session_state:
        st.subheader("Model Performance")
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = st.session_state['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        
        # Feature importance (for tree-based models)
        if st.session_state['model_type'] == "Random Forest":
            st.subheader("Feature Importance")
            fig_importance = utils.plot_feature_importance(
                st.session_state['model'], 
                st.session_state['feature_names']
            )
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Business insights
        st.subheader("Business Insights")
        insights = utils.generate_business_insights(
            df, st.session_state['model'], st.session_state['feature_names']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Key Findings:**")
            st.write(f"- Overall churn rate: {insights['overall_churn_rate']:.1%}")
            if 'highest_risk_contract' in insights:
                st.write(f"- Highest risk contract type: {insights['highest_risk_contract']} ({insights['highest_risk_contract_rate']:.1%})")
            if 'low_tenure_churn_rate' in insights:
                st.write(f"- New customers (≤12 months) churn rate: {insights['low_tenure_churn_rate']:.1%}")
        
        with col2:
            if 'top_churn_drivers' in insights:
                st.write("**Top Churn Drivers:**")
                for i, driver in enumerate(insights['top_churn_drivers'], 1):
                    st.write(f"{i}. {driver}")
        
        # Recommendations
        st.subheader("Business Recommendations")
        st.write("""
        **Retention Strategies:**
        - Focus on month-to-month contract customers with targeted offers
        - Implement early intervention programs for new customers (first 12 months)
        - Develop loyalty programs for high-value, long-tenure customers
        - Monitor and address the top churn drivers identified by the model
        - Consider personalized pricing strategies based on churn probability
        """)
    
    # Individual customer prediction
    st.header("Individual Customer Churn Prediction")
    
    if 'model' in st.session_state:
        st.write("Select a customer to predict their churn probability:")
        
        customer_idx = st.selectbox(
            "Customer ID",
            range(len(df)),
            format_func=lambda x: df.iloc[x]['customerID'] if 'customerID' in df.columns else f"Customer {x}"
        )
        
        if st.button("Predict Churn Probability"):
            # Get customer data
            customer_data = df.iloc[customer_idx:customer_idx+1].copy()
            
            # Encode categorical features
            customer_encoded = customer_data.copy()
            for col, encoder in st.session_state['encoders'].items():
                if col in customer_encoded.columns:
                    customer_encoded[col] = encoder.transform(customer_encoded[col])
            
            # Prepare features
            X_customer, _, _ = utils.prepare_features(customer_encoded)
            
            # Scale if needed
            if st.session_state['model_type'] == "Logistic Regression":
                X_customer = st.session_state['scaler'].transform(X_customer)
            
            # Predict
            churn_prob = st.session_state['model'].predict_proba(X_customer)[0, 1]
            churn_pred = st.session_state['model'].predict(X_customer)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Churn Probability", f"{churn_prob:.1%}")
                st.metric("Prediction", "Will Churn" if churn_pred else "Will Stay")
            
            with col2:
                # Risk level
                if churn_prob > 0.7:
                    risk_level = "High Risk"
                    risk_color = "red"
                elif churn_prob > 0.4:
                    risk_level = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_level = "Low Risk"
                    risk_color = "green"
                
                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
            
            # Customer details
            st.subheader("Customer Details")
            st.dataframe(customer_data, use_container_width=True)

else:
    st.info("Please upload a CSV file or use sample data to get started with the analysis.")
    
    st.markdown("""
    ### Expected Data Format
    
    Your CSV file should contain the following types of columns:
    - **Customer ID**: Unique identifier for each customer
    - **Demographics**: Age, gender, location, etc.
    - **Services**: Internet service, phone service, contract type, etc.
    - **Billing**: Monthly charges, total charges, payment method, etc.
    - **Churn**: Target variable (Yes/No or 1/0)
    
    ### Sample Data Structure
    ```
    customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,InternetService,Contract,PaymentMethod,MonthlyCharges,TotalCharges,Churn
    C0001,Female,0,Yes,No,1,No,DSL,Month-to-month,Electronic check,29.85,29.85,No
    C0002,Male,0,No,No,34,Yes,DSL,One year,Mailed check,56.95,1889.5,No
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This application is for educational purposes only. 
The predictions should not be used for actual business decisions without proper validation.
""")