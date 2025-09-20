"""
Customer Churn Prediction Library

This module contains functions for predicting customer churn using machine learning.
It includes data import, EDA, feature engineering, model training, and evaluation functions.

Author: Abdulhakeem Oyaqoob
Date: September 2024
"""

import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    Returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    """
    Perform eda on df and save figures to images folder

    input:
            df: pandas dataframe

    output:
            None
    """
    plt.figure(figsize=(20, 10))

    # Churn distribution
    plt.figure(figsize=(8, 6))
    df['Churn'].hist(bins=2, edgecolor='black')
    plt.title('Customer Churn Distribution')
    plt.xlabel('Churn (0: Existing, 1: Attrited)')
    plt.ylabel('Count')
    plt.xticks([0, 1])
    plt.savefig(
        './images/eda/churn_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # Customer age distribution
    plt.figure(figsize=(10, 6))
    df['Customer_Age'].hist(bins=30, edgecolor='black', alpha=0.7)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(
        './images/eda/customer_age_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # Marital status distribution
    plt.figure(figsize=(8, 6))
    df['Marital_Status'].value_counts().plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(
        './images/eda/marital_status_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # Total transaction amount distribution
    plt.figure(figsize=(10, 6))
    df['Total_Trans_Amt'].hist(bins=50, edgecolor='black', alpha=0.7)
    plt.title('Total Transaction Amount Distribution')
    plt.xlabel('Total Transaction Amount')
    plt.ylabel('Count')
    plt.savefig(
        './images/eda/total_trans_amt_distribution.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(20, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.savefig('./images/eda/heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def encoder_helper(df, category_lst, response):
    """
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns for
    """
    for col in category_lst:
        lst = []
        groups = df.groupby(col)[response].mean()

        for val in df[col]:
            lst.append(groups.loc[val])

        df[col + '_' + response] = lst

    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(df, cat_columns, response)

    y = df[response]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    plt.figure(figsize=(15, 8))

    # Random Forest results
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')

    # Logistic Regression results
    plt.text(0.5, 1.25, str('Logistic Regression Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.5, 0.05, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.5, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.5, 0.7, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')

    plt.axis('off')
    plt.savefig(
        './images/results/classification_report.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and stores the feature importances in pth

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, dpi=300, bbox_inches='tight')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    Train, store model results: images + scores, and store models

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # Use simplified parameters for faster training
    rfc = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    lrc = LogisticRegression(random_state=42, max_iter=3000)

    # Train models
    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Generate classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Generate feature importance plot
    feature_importance_plot(
        rfc,
        X_train,
        './images/results/feature_importance.png')

    # ROC curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8, name='Random Forest')
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8, name='Logistic Regression')
    plt.title('ROC Curves - Model Comparison')
    plt.legend()
    plt.savefig(
        './images/results/roc_curves.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # Save models with versioned names including date
    from datetime import datetime
    date_str = datetime.now().strftime('%Y%m%d')
    
    joblib.dump(rfc, f'./models/rfc_model_{date_str}.pkl')
    joblib.dump(lrc, f'./models/logistic_model_{date_str}.pkl')
    
    # Also save with standard names for backward compatibility
    joblib.dump(rfc, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    print(f"Models saved successfully:")
    print(f"- Random Forest: rfc_model_{date_str}.pkl")
    print(f"- Logistic Regression: logistic_model_{date_str}.pkl")


if __name__ == "__main__":
    # Load data
    df = import_data("./data/bank_data.csv")

    # Perform EDA
    perform_eda(df)

    # Feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    # Train models
    train_models(X_train, X_test, y_train, y_test)

    print("Churn prediction pipeline completed successfully!")
