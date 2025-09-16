"""
Utility functions for customer churn analysis.
Contains helper functions for data preprocessing, model evaluation, and visualization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def load_and_clean_data(file_path):
    """
    Load the customer data and fix any issues.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = pd.read_csv(file_path)
    
    # Fix the TotalCharges column (sometimes has spaces instead of numbers)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Convert Yes/No to 1/0 for machine learning
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df


def encode_categorical_features(df):
    """
    Convert text categories to numbers for ML models.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
        dict: Dictionary of label encoders
    """
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
    """
    Set up the data for training ML models.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: X (features), y (target), feature_names
    """
    # Remove columns we don't want to use for prediction
    features_to_drop = ['customerID']
    df_features = df.drop(columns=features_to_drop, errors='ignore')
    
    # Split into input features and what we want to predict
    X = df_features.drop('Churn', axis=1)
    y = df_features['Churn']
    
    return X, y, X.columns.tolist()


def evaluate_model(model, X_test, y_test):
    """
    Check how well the model performs.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics


def plot_churn_distribution(df):
    """
    Show how many customers churned vs stayed.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    churn_counts = df['Churn'].value_counts()
    churn_labels = ['No Churn', 'Churn']
    
    fig = go.Figure(data=[go.Pie(
        labels=churn_labels,
        values=churn_counts.values,
        hole=0.3,
        marker_colors=['#2E86AB', '#A23B72']
    )])
    
    fig.update_layout(
        title="Customer Churn Distribution",
        font=dict(size=14)
    )
    
    return fig


def plot_churn_by_feature(df, feature):
    """
    Show churn rates for different categories.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature name
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    churn_by_feature = df.groupby(feature)['Churn'].agg(['count', 'sum']).reset_index()
    churn_by_feature['churn_rate'] = churn_by_feature['sum'] / churn_by_feature['count']
    
    fig = px.bar(
        churn_by_feature,
        x=feature,
        y='churn_rate',
        title=f'Churn Rate by {feature}',
        labels={'churn_rate': 'Churn Rate'},
        color='churn_rate',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title='Churn Rate',
        font=dict(size=12)
    )
    
    return fig


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Show which features matter most for predictions.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
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
        title=f'Top {top_n} Feature Importances',
        labels={'importance': 'Importance', 'feature': 'Features'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        font=dict(size=12)
    )
    
    return fig


def generate_business_insights(df, model, feature_names):
    """
    Extract key business insights from the analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        model: Trained model
        feature_names (list): List of feature names
        
    Returns:
        dict: Dictionary of business insights
    """
    insights = {}
    
    # What percentage of customers churn overall
    insights['overall_churn_rate'] = df['Churn'].mean()
    
    # Find which customer groups are most likely to leave
    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn'].mean()
        insights['highest_risk_contract'] = contract_churn.idxmax()
        insights['highest_risk_contract_rate'] = contract_churn.max()
    
    if 'tenure' in df.columns:
        # New customers (first year) churn rate
        low_tenure_churn = df[df['tenure'] <= 12]['Churn'].mean()
        insights['low_tenure_churn_rate'] = low_tenure_churn
    
    # What factors drive churn the most
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        insights['top_churn_drivers'] = feature_importance.head(3)['feature'].tolist()
    
    return insights