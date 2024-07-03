## Folder Structure

### Automated Annotation/Labelled Data
Contains the original annotated data (training data) for different regions (UK and US).

- **UK/**
  - Contains the labelled data per issue and an aggregate of all issues.
  - `all_labelled.csv`: Aggregated labelled data for all issues.
  - `all_labelled_with_context.csv`: Aggregated labelled data with context.

- **US/**
  - Similar structure as the UK folder but for US data.

### Automated Annotation/Models
Contains scripts for various models to be used and tested to find the best fit for the data.

- Model scripts follow the same pipeline as the logistic regression model.

### Automated Annotation/Supplementary Scripts
Contains helper scripts for data preprocessing and aggregation.

- `preprocess_data.py`: Script to preprocess data and add context.
- `aggregate_training.py`: Script to aggregate training data for all issues.

## Pipeline Overview

### Step 1: Preprocess Data
Before training the models, the data needs to be preprocessed to add context to each data point. This is done using the `preprocess_data.py` script.

### Step 2: Aggregate Training Data
If working with multiple issues, aggregate the training data using the `aggregate_training.py` script.

### Step 3: Train and Test Models
Use the model scripts in the Models folder to train and test different models. The pipeline involves:
- Loading the preprocessed data.
- Performing TF-IDF vectorization.
- Training the model with cross-validation.
- Testing it on the test data.

## Detailed Description

### Automated Annotation/Labelled Data

#### Structure

- **UK/**:
  - Contains labelled data for various issues and aggregated data.
  - `pro_brexit_labelled.csv`: Labelled data for the pro-Brexit stance.
  - `anti_brexit_labelled.csv`: Labelled data for the anti-Brexit stance.
  - `all_labelled.csv`: Aggregated labelled data for all issues.
  - `all_labelled_with_context.csv`: Aggregated labelled data with context added.

- **US/**:
  - Similar structure as the UK folder but for US data.

### Automated Annotation/Models

#### Structure

- `logistic_regression_model.py`: Script for training and testing a logistic regression model.
- Additional model scripts (e.g., `svm_model.py`, `random_forest_model.py`) follow a similar structure to the logistic regression model script.

#### Usage

1. **Load Data**: Load the preprocessed data using pandas.
2. **Filter Training Data from Test Data**: Ensure that the training data IDs are removed from the test data.
3. **TF-IDF Vectorization**: Convert text data into TF-IDF vectors.
4. **Cross-Validation and Model Training**: Perform cross-validation and train the model.
5. **Predict and Resolve Contradictions**: Make predictions on the test data and resolve any contradictions in the results.

### Automated Annotation/Supplementary Scripts

#### Structure

- `preprocess_data.py`: Preprocesses the data to add context. It loads the data, combines the title and body, builds the context for each data point, and saves the preprocessed data.
- `aggregate_training.py`: Aggregates the training data for all issues into a single file.
