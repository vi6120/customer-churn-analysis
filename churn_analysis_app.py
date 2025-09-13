"""
Complete Customer Churn Analysis Application
Single file containing all functionality for Streamlit deployment.
Educational Purpose Only.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility Functions
@st.cache_data
def generate_sample_data(n_samples=2000):
    """Generate realistic sample customer data."""
    np.random.seed(42)
    
    data = {
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
    
    # Create realistic churn patterns
    churn_prob = np.zeros(n_samples)
    for i in range(n_samples):
        prob = 0.1
        if data['Contract'][i] == 'Month-to-month':
            prob += 0.3
        elif data['Contract'][i] == 'One year':
            prob += 0.1
        if data['tenure'][i] <= 12:
            prob += 0.2
        if data['MonthlyCharges'][i] > 80:
            prob += 0.15
        if data['InternetService'][i] == 'Fiber optic':
            prob += 0.1
        if data['PaymentMethod'][i] == 'Electronic check':
            prob += 0.1
        churn_prob[i] = min(prob, 0.8)
    
    data['Churn'] = np.random.binomial(1, churn_prob, n_samples)
    return pd.DataFrame(data)

def clean_data(df):
    """Clean and preprocess data."""
    df = df.copy()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

def encode_features(df):
    """Encode categorical features."""
    df_encoded = df.copy()
    encoders = {}
    
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns.drop('customerID', errors='ignore')
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    
    return df_encoded, encoders

def prepare_features(df):
    """Prepare features for ML models."""
    features_to_drop = ['customerID']
    df_features = df.drop(columns=features_to_drop, errors='ignore')
    X = df_features.drop('Churn', axis=1)
    y = df_features['Churn']
    return X, y, X.columns.tolist()

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

def plot_churn_distribution(df):
    """Plot churn distribution."""
    churn_counts = df['Churn'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=['No Churn', 'Churn'],
        values=churn_counts.values,
        hole=0.3,
        marker_colors=['#2E86AB', '#A23B72']
    )])
    fig.update_layout(title="Customer Churn Distribution")
    return fig

def plot_churn_by_feature(df, feature):
    """Plot churn rate by feature."""
    churn_by_feature = df.groupby(feature)['Churn'].agg(['count', 'sum']).reset_index()
    churn_by_feature['churn_rate'] = churn_by_feature['sum'] / churn_by_feature['count']
    
    fig = px.bar(
        churn_by_feature,
        x=feature,
        y='churn_rate',
        title=f'Churn Rate by {feature}',
        color='churn_rate',
        color_continuous_scale='Reds'
    )
    return fig

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance."""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Feature Importances'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def calculate_churn_kpis(df, model, feature_names):
    """Calculate advanced churn KPIs including financial impact."""
    kpis = {}
    
    if not hasattr(model, 'feature_importances_'):
        return kpis
    
    # Get top 3 features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_3_features = importance_df.head(3)['feature'].tolist()
    kpis['top_3_features'] = top_3_features
    
    # Calculate financial impact if TotalCharges exists
    if 'TotalCharges' in df.columns:
        churned_customers = df[df['Churn'] == 1]
        total_charges_lost = churned_customers['TotalCharges'].sum()
        total_charges_all = df['TotalCharges'].sum()
        
        kpis['total_charges_lost'] = total_charges_lost
        kpis['total_charges_all'] = total_charges_all
        kpis['churn_loss_percentage'] = (total_charges_lost / total_charges_all) * 100
        
        # Calculate impact by top features (approximate)
        feature_impact = {}
        for feature in top_3_features:
            if feature in df.columns:
                # For categorical features, find the category with highest churn
                if df[feature].dtype == 'object' or df[feature].nunique() < 10:
                    feature_churn = df.groupby(feature)['Churn'].agg(['sum', 'count']).reset_index()
                    feature_churn['churn_rate'] = feature_churn['sum'] / feature_churn['count']
                    worst_category = feature_churn.loc[feature_churn['churn_rate'].idxmax(), feature]
                    
                    worst_category_customers = df[(df[feature] == worst_category) & (df['Churn'] == 1)]
                    feature_loss = worst_category_customers['TotalCharges'].sum()
                else:
                    # For numerical features, use median split
                    median_val = df[feature].median()
                    high_risk_customers = df[(df[feature] > median_val) & (df['Churn'] == 1)]
                    feature_loss = high_risk_customers['TotalCharges'].sum()
                
                feature_impact[feature] = feature_loss
        
        kpis['feature_impact'] = feature_impact
        kpis['top_3_total_loss'] = sum(feature_impact.values())
        
        if total_charges_all > 0:
            kpis['top_3_loss_percentage'] = (kpis['top_3_total_loss'] / total_charges_all) * 100
    
    return kpis

# Main Application
def main():
    # Custom CSS for responsive design and centered title
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        color: #666;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
        }
        .subtitle {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Centered title
    st.markdown('<h1 class="main-title">Customer Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Professional machine learning solution for customer retention analysis<br><strong>Educational Purpose Only</strong>: This tool is for learning and demonstration purposes.</p>', unsafe_allow_html=True)
    
    # Professional header info
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #1f77b4;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>Author:</strong> Vikas Ramaswamy | <strong>Version:</strong> 1.0 | <strong>Technology:</strong> Python, Scikit-learn, XGBoost, Streamlit
            </div>
            <div style="color: #6c757d; font-size: 0.9rem;">
                Professional Machine Learning Solution for Customer Analytics
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    st.sidebar.caption("Version 1.0.0")
    st.sidebar.info("Sample data is loaded by default for quick exploration")
    
    # Quick links
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Feedback & Support:**")
    st.sidebar.markdown("- [Report Bug](https://github.com/vi6120/customer-churn-analysis/issues/new?labels=bug&title=[BUG]%20Issue%20Title)")
    st.sidebar.markdown("- [Request Feature](https://github.com/vi6120/customer-churn-analysis/issues/new?labels=enhancement&title=[FEATURE]%20Feature%20Title)")
    st.sidebar.markdown("- [Documentation](https://github.com/vi6120/customer-churn-analysis#readme)")
    
    # Mobile view toggle
    mobile_view = st.sidebar.checkbox("Mobile Layout", help="Optimize layout for mobile screens")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Use Sample Data", "Upload CSV File"]
    )
    
    # Load data
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Using uploaded data")
        else:
            st.info("Please upload a CSV file or switch to sample data to get started.")
            return
    else:
        df = generate_sample_data()
        st.sidebar.success("Using sample data (2000 customers)")
    
    # Clean data
    df = clean_data(df)
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "XGBoost"]
    )
    
    # Main content
    if df is not None:
        # Dataset overview
        st.header("Dataset Overview")
        # Responsive columns
        if mobile_view:
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
        else:
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
        
        # EDA Section
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_churn = plot_churn_distribution(df)
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            if 'customerID' in categorical_features:
                categorical_features.remove('customerID')
            
            if categorical_features:
                selected_feature = st.selectbox("Analyze Churn by Feature", categorical_features)
                fig_feature = plot_churn_by_feature(df, selected_feature)
                st.plotly_chart(fig_feature, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            fig_corr = px.imshow(
                numeric_df.corr(),
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Machine Learning Section
        st.header("Machine Learning Model")
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Prepare data
                df_encoded, encoders = encode_features(df)
                X, y, feature_names = prepare_features(df_encoded)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train selected model
                if model_type == "Logistic Regression":
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)

                    X_test_scaled = scaler.transform(X_test)
                    model = LogisticRegression(random_state=42)
                    model.fit(X_train_scaled, y_train)
                    X_test_for_eval = X_test_scaled
                    st.session_state['scaler'] = scaler
                elif model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    X_test_for_eval = X_test
                else:  # XGBoost
                    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                    model.fit(X_train, y_train)
                    X_test_for_eval = X_test
                
                # Evaluate model
                metrics = evaluate_model(model, X_test_for_eval, y_test)
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['feature_names'] = feature_names
                st.session_state['encoders'] = encoders
                st.session_state['metrics'] = metrics
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['model_type'] = model_type
                st.session_state['df'] = df
            
            st.success("Model trained successfully!")
        
        # Display results if model is trained
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
            
            # Financial Impact KPIs (only show if TotalCharges exists)
            if 'TotalCharges' in df.columns:
                st.subheader("Financial Impact Analysis")
                
                kpis = calculate_churn_kpis(df, st.session_state['model'], st.session_state['feature_names'])
                
                if mobile_view:
                    col1, col2 = st.columns(2)
                    col3 = st.columns(1)[0]
                else:
                    col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'total_charges_lost' in kpis:
                        st.metric(
                            "Total Revenue Lost", 
                            f"${kpis['total_charges_lost']:,.0f}",
                            help="Total charges lost due to customer churn"
                        )
                
                with col2:
                    if 'churn_loss_percentage' in kpis:
                        st.metric(
                            "Revenue Loss %", 
                            f"{kpis['churn_loss_percentage']:.1f}%",
                            help="Percentage of total revenue lost to churn"
                        )
                
                with col3:
                    if 'top_3_loss_percentage' in kpis:
                        st.metric(
                            "Top 3 Features Impact", 
                            f"{kpis['top_3_loss_percentage']:.1f}%",
                            help="Revenue loss percentage attributed to top 3 churn drivers"
                        )
                
                # Show top 3 features breakdown
                if 'top_3_features' in kpis and 'feature_impact' in kpis:
                    st.subheader("Top 3 Churn Drivers Financial Impact")
                    
                    impact_data = []
                    for i, feature in enumerate(kpis['top_3_features'], 1):
                        loss = kpis['feature_impact'].get(feature, 0)
                        impact_data.append({
                            'Rank': f"#{i}",
                            'Feature': feature,
                            'Revenue Lost': f"${loss:,.0f}",
                            'Impact %': f"{(loss/kpis['total_charges_all']*100):.1f}%" if kpis['total_charges_all'] > 0 else "0%"
                        })
                    
                    impact_df = pd.DataFrame(impact_data)
                    st.dataframe(impact_df, use_container_width=True, hide_index=True)
            
            # Feature importance
            if st.session_state['model_type'] in ["Random Forest", "XGBoost"]:
                st.subheader("Feature Importance")
                fig_importance = plot_feature_importance(
                    st.session_state['model'], 
                    st.session_state['feature_names']
                )
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            # Business insights
            st.subheader("Business Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Findings:**")
                st.write(f"- Overall churn rate: {df['Churn'].mean():.1%}")
                if 'Contract' in df.columns:
                    contract_churn = df.groupby('Contract')['Churn'].mean()
                    highest_risk = contract_churn.idxmax()
                    st.write(f"- Highest risk contract: {highest_risk} ({contract_churn.max():.1%})")
                if 'tenure' in df.columns:
                    new_customer_churn = df[df['tenure'] <= 12]['Churn'].mean()
                    st.write(f"- New customers (≤12 months): {new_customer_churn:.1%}")
            
            with col2:
                if hasattr(st.session_state['model'], 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': st.session_state['feature_names'],
                        'importance': st.session_state['model'].feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.write("**Top Churn Drivers:**")
                    for i, (_, row) in enumerate(importance_df.head(3).iterrows(), 1):
                        st.write(f"{i}. {row['feature']}")
            
            # Recommendations
            st.subheader("Business Recommendations")
            st.write("""
            **Retention Strategies:**
            - Focus on month-to-month contract customers with targeted offers
            - Implement early intervention programs for new customers
            - Develop loyalty programs for high-value customers
            - Monitor top churn drivers identified by the model
            - Consider personalized pricing based on churn probability
            """)
            
            # Individual prediction
            st.header("Individual Customer Prediction")
            
            customer_idx = st.selectbox(
                "Select Customer",
                range(len(df)),
                format_func=lambda x: df.iloc[x]['customerID'] if 'customerID' in df.columns else f"Customer {x}"
            )
            
            if st.button("Predict Churn Probability"):
                # Get customer data
                customer_data = df.iloc[customer_idx:customer_idx+1].copy()
                
                # Encode
                customer_encoded = customer_data.copy()
                for col, encoder in st.session_state['encoders'].items():
                    if col in customer_encoded.columns:
                        customer_encoded[col] = encoder.transform(customer_encoded[col])
                
                # Prepare features
                X_customer = customer_encoded.drop(['customerID', 'Churn'], axis=1, errors='ignore')
                
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
                    if churn_prob > 0.7:
                        risk_level = "High Risk"
                        risk_color = "red"
                    elif churn_prob > 0.4:
                        risk_level = "Medium Risk"
                        risk_color = "orange"
                    else:
                        risk_level = "Low Risk"
                        risk_color = "green"
                    
                    if risk_color == "red":
                        st.error(f"Risk Level: {risk_level}")
                    elif risk_color == "orange":
                        st.warning(f"Risk Level: {risk_level}")
                    else:
                        st.success(f"Risk Level: {risk_level}")
                
                # Customer details
                st.subheader("Customer Details")
                st.dataframe(customer_data, use_container_width=True)

    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
        <div style="margin-bottom: 1rem;">
            <strong>Customer Churn Analysis Dashboard</strong>
        </div>
        <div style="color: #6c757d; margin-bottom: 1rem;">
            Professional machine learning solution for predicting customer churn and developing retention strategies
        </div>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <div><strong>Models:</strong> Logistic Regression | Random Forest | XGBoost</div>
            <div><strong>Technology:</strong> Python | Scikit-learn | Streamlit</div>
        </div>
        <div style="color: #6c757d; font-size: 0.9rem;">
            © 2024 Vikas Ramaswamy | Professional Analytics Portfolio | Educational Purpose Only
        </div>
    </div>
    """)

if __name__ == "__main__":
    main()