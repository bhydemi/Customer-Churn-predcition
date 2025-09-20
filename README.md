# Credit Card Customer Churn Prediction

This project implements a machine learning solution to identify credit card customers who are most likely to churn. The solution follows software engineering best practices including modular code design, comprehensive testing, logging, and proper documentation.

## Project Description

Customer churn prediction is a critical business problem for financial institutions. This project provides a complete machine learning pipeline that:

- Performs exploratory data analysis (EDA)
- Engineers relevant features from customer data
- Trains multiple machine learning models (Random Forest and Logistic Regression)
- Evaluates model performance using various metrics
- Generates visualizations for model interpretation

## Project Structure

```
.
├── churn_library.py                    # Main library with ML functions
├── churn_script_logging_and_tests.py   # Testing and logging script
├── churn_notebook.ipynb                # Jupyter notebook with complete workflow
├── README.md                           # Project documentation
├── data/
│   └── bank_data.csv                   # Customer data
├── images/
│   ├── eda/                           # Exploratory data analysis plots
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── marital_status_distribution.png
│   │   ├── total_trans_amt_distribution.png
│   │   └── heatmap.png
│   └── results/                       # Model results and plots
│       ├── classification_report.png
│       ├── feature_importance.png
│       └── roc_curves.png
├── logs/
│   └── churn_library.log              # Test execution logs
└── models/
    ├── logistic_model.pkl             # Trained logistic regression model
    └── rfc_model.pkl                  # Trained random forest model
```

## Files and Data Description

### Main Files

- **churn_library.py**: Contains all the core functions for the ML pipeline
  - `import_data()`: Loads and preprocesses the data
  - `perform_eda()`: Generates exploratory data analysis plots
  - `encoder_helper()`: Encodes categorical variables
  - `perform_feature_engineering()`: Prepares features and splits data
  - `train_models()`: Trains and evaluates ML models
  - `classification_report_image()`: Creates classification report visualizations
  - `feature_importance_plot()`: Generates feature importance plots

- **churn_script_logging_and_tests.py**: Comprehensive testing suite with logging
  - Tests each function in churn_library.py
  - Logs test results and errors
  - Validates output files and directories

- **churn_notebook.ipynb**: Interactive Jupyter notebook
  - Step-by-step walkthrough of the ML process
  - Detailed explanations and visualizations
  - Can be used for experimentation and analysis

### Data Description

The dataset (`bank_data.csv`) contains customer information including:

- **Demographics**: Age, Gender, Education Level, Marital Status
- **Financial**: Income Category, Credit Limit, Total Revolving Balance
- **Behavioral**: Transaction amounts, transaction counts, months inactive
- **Target**: Attrition Flag (converted to binary Churn variable)

### Key Features Used

- Customer demographics and financial information
- Transaction patterns and history
- Account relationship metrics
- Encoded categorical variables using mean target encoding

## Dependencies

The project requires the following Python packages:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Running the Files

### Method 1: Run the Complete Pipeline

Execute the main library script to run the entire ML pipeline:

```bash
python churn_library.py
```

This will:
1. Load and process the data
2. Generate EDA visualizations
3. Train ML models
4. Create result plots and save models

### Method 2: Interactive Exploration

Use the Jupyter notebook for interactive analysis:

```bash
jupyter notebook churn_notebook.ipynb
```

### Method 3: Testing and Validation

Run the test suite to validate all functions and generate logs:

```bash
python churn_script_logging_and_tests.py
```

This will:
- Test each function in churn_library.py
- Validate that outputs are generated correctly
- Create detailed logs in `./logs/churn_library.log`

## Code Quality

The project follows PEP 8 coding standards and includes:

### Style Checking

Use autopep8 for automatic formatting:
```bash
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
```

Use pylint for code analysis:
```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

Target: Pylint score > 7.0 for both files

### Documentation

- All functions have comprehensive docstrings
- Clear parameter and return value descriptions
- Module-level documentation with author and date information

## Model Performance

The solution implements two machine learning models:

1. **Random Forest Classifier**
   - Uses GridSearchCV for hyperparameter optimization
   - Generally provides better performance for this dataset
   - Offers feature importance insights

2. **Logistic Regression**
   - Provides a baseline linear model
   - Faster training and prediction
   - More interpretable coefficients

### Evaluation Metrics

- Classification reports (precision, recall, F1-score)
- ROC curves and AUC scores
- Feature importance rankings
- Confusion matrices

## Output Files

After running the pipeline, the following files will be generated:

### EDA Visualizations (`./images/eda/`)
- Customer churn distribution
- Age and demographic distributions
- Correlation heatmap
- Transaction pattern analysis

### Model Results (`./images/results/`)
- Classification performance reports
- ROC curves comparing models
- Feature importance plots

### Trained Models (`./models/`)
- Random Forest model (`.pkl` format)
- Logistic Regression model (`.pkl` format)

### Logs (`./logs/`)
- Detailed test execution logs
- Error tracking and debugging information

## Usage Examples

### Loading and Using Trained Models

```python
import joblib
import pandas as pd

# Load trained models
rf_model = joblib.load('./models/rfc_model.pkl')
lr_model = joblib.load('./models/logistic_model.pkl')

# Make predictions on new data
# (assuming new_data is preprocessed similarly to training data)
rf_predictions = rf_model.predict(new_data)
lr_predictions = lr_model.predict(new_data)
```

### Custom Analysis

```python
import churn_library as cls

# Load data
df = cls.import_data('./data/bank_data.csv')

# Perform custom EDA
cls.perform_eda(df)

# Feature engineering
X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')

# Train models
cls.train_models(X_train, X_test, y_train, y_test)
```

## Testing Framework

The testing suite validates:

- Data loading and preprocessing
- EDA plot generation
- Feature engineering correctness
- Model training and saving
- Result visualization creation
- Error handling and edge cases

All tests include comprehensive logging for debugging and validation.

## Contributing

To contribute to this project:

1. Follow PEP 8 coding standards
2. Add comprehensive docstrings to all functions
3. Include unit tests for new functionality
4. Update documentation as needed
5. Ensure pylint score remains above 7.0

## License

This project is for educational and demonstration purposes.

## Author

**Abdulhakeem Oyaqoob**  
*September 2024*

---

For questions or issues, please refer to the logged outputs in `./logs/churn_library.log` for detailed debugging information.