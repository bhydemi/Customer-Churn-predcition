"""
Testing and Logging for Churn Library

This module contains unit tests and logging for the churn prediction library functions.
It tests each function in churn_library.py and logs the results.

Author: Abdulhakeem Oyaqoob
Date: September 2024
"""

import os
import logging
import glob
import pandas as pd
import pytest
import churn_library as cls

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    Test data import - this example is completed for you to assist with the other test functions

    input:
            import_data: function to test

    output:
            None
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: DataFrame has rows and columns")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        assert 'Churn' in df.columns
        logging.info("Testing import_data: Churn column created successfully")
    except AssertionError as err:
        logging.error("Testing import_data: Churn column not found")
        raise err

    return df


def test_eda(perform_eda):
    """
    Test perform eda function

    input:
            perform_eda: function to test

    output:
            None
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: ERROR - %s", str(err))
        raise err

    # Check if EDA images were created
    try:
        assert os.path.exists('./images/eda/churn_distribution.png')
        assert os.path.exists('./images/eda/customer_age_distribution.png')
        assert os.path.exists('./images/eda/marital_status_distribution.png')
        assert os.path.exists('./images/eda/total_trans_amt_distribution.png')
        assert os.path.exists('./images/eda/heatmap.png')
        logging.info(
            "Testing perform_eda: All EDA images created successfully")
    except AssertionError as err:
        logging.error("Testing perform_eda: EDA images not found")
        raise err


def test_encoder_helper(encoder_helper):
    """
    Test encoder helper function

    input:
            encoder_helper: function to test

    output:
            None
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df_encoded = encoder_helper(df, cat_columns, 'Churn')
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: ERROR - %s", str(err))
        raise err

    # Check if encoded columns were created
    try:
        for col in cat_columns:
            assert f"{col}_Churn" in df_encoded.columns
        logging.info(
            "Testing encoder_helper: All encoded columns created successfully")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Encoded columns not found")
        raise err

    # Check if encoded values are numeric
    try:
        for col in cat_columns:
            assert pd.api.types.is_numeric_dtype(df_encoded[f"{col}_Churn"])
        logging.info("Testing encoder_helper: All encoded columns are numeric")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Encoded columns are not numeric")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering function

    input:
            perform_feature_engineering: function to test

    output:
            None
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, 'Churn')
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - %s",
            str(err))
        raise err

    # Check if the splits have the correct shapes
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            "Testing perform_feature_engineering: Train/test splits have correct shapes")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Train/test splits have incorrect shapes")
        raise err

    # Check if feature matrix and target vector have consistent dimensions
    try:
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        logging.info(
            "Testing perform_feature_engineering: Feature matrix and target vector dimensions are consistent")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Inconsistent dimensions")
        raise err


def test_train_models(train_models):
    """
    Test train_models function

    input:
            train_models: function to test

    output:
            None
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, 'Churn')
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: ERROR - %s", str(err))
        raise err

    # Check if models were saved
    try:
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
        logging.info("Testing train_models: Models saved successfully")
    except AssertionError as err:
        logging.error("Testing train_models: Models not saved")
        raise err

    # Check if result images were created
    try:
        assert os.path.exists('./images/results/classification_report.png')
        assert os.path.exists('./images/results/feature_importance.png')
        assert os.path.exists('./images/results/roc_curves.png')
        logging.info(
            "Testing train_models: Result images created successfully")
    except AssertionError as err:
        logging.error("Testing train_models: Result images not created")
        raise err


def test_classification_report_image(classification_report_image):
    """
    Test classification_report_image function

    input:
            classification_report_image: function to test

    output:
            None
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, 'Churn')

        # Create dummy predictions for testing
        y_train_preds_lr = [0] * len(y_train)
        y_train_preds_rf = [0] * len(y_train)
        y_test_preds_lr = [0] * len(y_test)
        y_test_preds_rf = [0] * len(y_test)

        classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf)
        logging.info("Testing classification_report_image: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing classification_report_image: ERROR - %s",
            str(err))
        raise err

    # Check if classification report image was created
    try:
        assert os.path.exists('./images/results/classification_report.png')
        logging.info(
            "Testing classification_report_image: Classification report image created successfully")
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: Classification report image not created")
        raise err


def test_feature_importance_plot(feature_importance_plot):
    """
    Test feature_importance_plot function

    input:
            feature_importance_plot: function to test

    output:
            None
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, 'Churn')

        # Train a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)

        feature_importance_plot(
            model, X_train, './images/results/test_feature_importance.png')
        logging.info("Testing feature_importance_plot: SUCCESS")
    except Exception as err:
        logging.error("Testing feature_importance_plot: ERROR - %s", str(err))
        raise err

    # Check if feature importance plot was created
    try:
        assert os.path.exists('./images/results/test_feature_importance.png')
        logging.info(
            "Testing feature_importance_plot: Feature importance plot created successfully")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: Feature importance plot not created")
        raise err


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./images/eda', exist_ok=True)
    os.makedirs('./images/results', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    # Run tests
    logging.info("Starting churn library tests...")

    # Test import_data
    df = test_import(cls.import_data)

    # Test perform_eda
    test_eda(cls.perform_eda)

    # Test encoder_helper
    test_encoder_helper(cls.encoder_helper)

    # Test perform_feature_engineering
    test_perform_feature_engineering(cls.perform_feature_engineering)

    # Test classification_report_image
    test_classification_report_image(cls.classification_report_image)

    # Test feature_importance_plot
    test_feature_importance_plot(cls.feature_importance_plot)

    # Test train_models (this should be last as it takes longest)
    test_train_models(cls.train_models)

    logging.info("All tests completed successfully!")
    print("All tests passed! Check ./logs/churn_library.log for detailed results.")
