import os
import numpy as np
import pandas as pd
import pickle as pk
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from aif360.sklearn.datasets import fetch_compas
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def get_FGCE_Directory():
    """Get the path of the 'FGCE' directory."""
    current_dir = os.getcwd()
    while os.path.basename(current_dir) != 'FGCE':
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            return None
    return current_dir


FGCE_DIR = get_FGCE_Directory()


def load_dataset(datasetName='Student'):
    if (datasetName == "Student"):
        return load_student()
    if (datasetName == "Adult"):
        return load_adult()
    if (datasetName == "Compas"):
        return load_compas()
    if (datasetName == "Heloc"):
        return load_heloc()
    if (datasetName == "GermanCredit"):
        return load_german_credit()
    if (datasetName == "Compas_full_data"):
        return load_compas_full_data()

    return pd.DataFrame([]), [], [], None, None, None, None, []


def load_sample_data():
    data = pd.DataFrame({
        'Age': ['22', '23', '24', '25'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Income': ['High', 'Low', 'Medium', 'High'],
        'Score': [12, 15, 13, 12]
    })
    return data


def load_compas_full_data():
    data = pd.read_csv('data/Compas_full_data.csv')

    data = data.drop(
        columns=['id', 'name', 'first', 'last', 'age_cat', 'c_case_number', 'c_arrest_date', 'c_charge_desc',
                 'r_case_number', 'r_charge_degree', 'c_offense_date', 'compas_screening_date', 'score_text',
                 # 'v_score_text',
                 'dob', 'screening_date', 'start', 'end', 'event', 'type_of_assessment', 'v_type_of_assessment',
                 'v_screening_date', 'decile_score.1', 'priors_count.1', 'is_recid', 'vr_charge_degree',
                 # 'v_decile_score', 'vr_charge_degree'
                 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid',
                 'vr_case_number', 'vr_offense_date', 'vr_charge_desc'])
    data = data.dropna()
    data['c_jail_in'] = pd.to_datetime(data['c_jail_in'])
    data['c_jail_out'] = pd.to_datetime(data['c_jail_out'])
    data['time_in_jail'] = (data['c_jail_out'] - data['c_jail_in']).dt.days
    data = data.drop(columns=['c_jail_in', 'c_jail_out'])

    data['in_custody'] = pd.to_datetime(data['in_custody'])
    data['out_custody'] = pd.to_datetime(data['out_custody'])
    data['time_in_custody'] = (data['out_custody'] - data['in_custody']).dt.days
    data = data.drop(columns=['in_custody', 'out_custody'])

    data = data.drop_duplicates()
    data = data.reset_index(drop=True)

    continuous_features = ['days_b_screening_arrest', 'c_days_from_compas', 'time_in_custody',
                           'time_in_jail']  # , 'age'

    # print(data[data.duplicated()].shape)
    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features)
    # print(data[data.duplicated()].shape)
    data = data[[col for col in data.columns if col != 'two_year_recid'] + ['two_year_recid']]

    data = data.reset_index(drop=True)
    data_df_copy = data.copy()
    data_df_copy = data_df_copy[data_df_copy.columns[:-1]]

    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)
    # print(data[data.duplicated()].shape)
    data = data.drop_duplicates()
    # print(data[data.duplicated()].shape)

    FEATURE_COLUMNS = data.columns[:-1]
    TARGET_COLUMNS = 'two_year_recid'

    data = data.reset_index(drop=True)
    data_df_copy = data_df_copy.reset_index(drop=True)

    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, continuous_features


def load_compas():
    X, y = fetch_compas()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    TARGET_COLUMNS = 'two_year_recid'
    data = X
    # Drop the columns 'c_charge_desc' and 'age' from the DataFrame
    data = data.drop(['c_charge_desc', 'age_cat'], axis=1)

    # reset the index

    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])

    data_df_copy = data.copy()

    # make y  into a dataframe
    y = pd.DataFrame(y, columns=[TARGET_COLUMNS])
    y, _, _ = preprocess_dataset(y, continuous_features=[])

    data[TARGET_COLUMNS] = y

    ### Scale the dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    FEATURE_COLUMNS = data.columns[:-1]

    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, []


def load_german_credit():
    data_df = pd.read_csv(data_df=pd.read_csv(f"{FGCE_DIR}/data/GermanCredit.csv"))
    # Drop the target column
    TARGET_COLUMNS = data_df.columns[-1]
    data = data_df.drop(columns=[TARGET_COLUMNS])

    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])

    ### Scale the dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    FEATURE_COLUMNS = data.columns

    ### Add the target column back
    data[TARGET_COLUMNS] = data_df[TARGET_COLUMNS]

    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, []


def load_student():
    data_df = pd.read_csv(f"{FGCE_DIR}/data/student.csv")

    # Drop the target column
    TARGET_COLUMNS = data_df.columns[-1]
    data = data_df.drop(columns=[TARGET_COLUMNS])

    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])

    data_df_copy = data.copy()

    # ### Scale the dataset
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data_scaled = min_max_scaler.fit_transform(data)
    # data = pd.DataFrame(data_scaled, columns=data.columns)

    FEATURE_COLUMNS = data.columns

    ### Add the target column back
    data[TARGET_COLUMNS] = data_df[TARGET_COLUMNS]

    ### Scale the dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, []


def load_adult():
    data_df = pd.read_csv(f"adult.csv")
    # Drop the target column
    TARGET_COLUMNS = data_df.columns[-1]
    data = data_df.drop(columns=[TARGET_COLUMNS])

    data, numeric_columns, categorical_columns = preprocess_dataset(data, continuous_features=[])
    data_df_copy = data.copy()
    # Scale the dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    FEATURE_COLUMNS = data.columns

    # Add the target column back
    data[TARGET_COLUMNS] = data_df[TARGET_COLUMNS]

    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, []


def load_heloc():
    data_df = pd.read_csv(f"heloc.csv")
    # data_df = data_df.dropna()

    # data_df = data_df[(data_df.iloc[:, 1:]>=0).any(axis=1)]

    data_df = data_df[(data_df.iloc[:, 1:] >= 0).all(axis=1)]

    # reset the index
    data_df = data_df.reset_index(drop=True)

    data_df_copy = data_df.copy()

    first_column = data_df.pop(data_df.columns[0])

    FEATURE_COLUMNS = data_df.columns

    # Insert the first column as the last column
    data_df.insert(len(data_df.columns), first_column.name, first_column)

    # Drop the target column
    TARGET_COLUMNS = "RiskPerformance"

    continuous_featues = ['MSinceOldestTradeOpen',
                          'AverageMInFile',
                          'NetFractionInstallBurden',
                          'NetFractionRevolvingBurden',
                          'MSinceMostRecentTradeOpen',
                          'PercentInstallTrades',
                          'PercentTradesWBalance',
                          'NumTotalTrades',
                          'MSinceMostRecentDelq',
                          'NumSatisfactoryTrades',
                          'PercentTradesNeverDelq',
                          'ExternalRiskEstimate']

    data, numeric_columns, categorical_columns = preprocess_dataset(data_df, continuous_features=continuous_featues)

    data_df_copy = data.copy()

    ### Scale the dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    return data, FEATURE_COLUMNS, TARGET_COLUMNS, numeric_columns, categorical_columns, min_max_scaler, data_df_copy, continuous_featues


def calculate_num_bins(num_unique_values, value_range):
    # Calculate the number of bins using a heuristic approach
    num_bins = min(6, int(np.log2(num_unique_values)) + 1)
    # Limit the number of bins based on the range of values
    num_bins = min(num_bins, value_range)
    return num_bins


def preprocess_dataset(df, continuous_features=[]):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()

    numeric_columns = []
    categorical_columns = []

    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Check if the column is categorical
        if df[col].dtype == 'object' or df[col].dtype == 'category' and col not in continuous_features:
            categorical_columns.append(col)
            # If the column has only two unique values, treat it as binary categorical
            if len(df[col].unique()) == 2:
                # Label encode binary categorical features
                df[col] = label_encoder.fit_transform(df[col])
            else:
                # One-hot encode regular categorical features
                encoded_values = onehot_encoder.fit_transform(df[[col]])
                # Create new column names for the one-hot encoded features
                new_cols = [col + '_' + str(i) for i in range(encoded_values.shape[1])]
                # Convert the encoded values to a DataFrame and assign column names
                encoded_df = pd.DataFrame(encoded_values.toarray(), columns=new_cols)
                # Concatenate the encoded DataFrame with the original DataFrame
                df = pd.concat([df, encoded_df], axis=1)
                # Drop the original categorical column from the DataFrame
                df.drop(col, axis=1, inplace=True)
        # If the column is numerical but in string format and not in continuous_features, convert it to numerical type
        elif df[col].dtype == 'object' or df[col].dtype == 'category' and df[
            col].str.isnumeric().all() and col not in continuous_features:
            df[col] = df[col].astype(int)  # Convert to integer type
            categorical_columns.append(col)
        # If the column is a continuous feature, discretize it into bins
        elif col in continuous_features:
            numeric_columns.append(col)
            # Calculate the number of bins
            num_unique_values = len(df[col].unique())
            value_range = df[col].max() - df[col].min()
            num_bins = calculate_num_bins(num_unique_values, value_range)

            # Discretize into bins
            bin_discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
            bins = bin_discretizer.fit_transform(df[[col]])
            # Replace the original continuous feature with the binned values
            df[col] = bins.astype(int)
        else:
            # Here are numerical columns. If the column has only 2 unique values, dont add it to numeric_columns
            if len(df[col].unique()) > 2:
                numeric_columns.append(col)
    return df, numeric_columns, categorical_columns


# def calculate_confusion_matrix_metric(data, features, target, classifier_type='logistic_regression', metric='accuracy'):
#     """
#     Train a classifier, predict on test data, and calculate a specific metric from the confusion matrix.
#
#     :param data: DataFrame containing the dataset
#     :param features: List of feature column names
#     :param target: Name of the target column
#     :param classifier_type: Type of classifier ('logistic_regression', 'svm', 'random_forest', 'naive_bayes')
#     :param metric: Metric to calculate ('accuracy', 'precision', 'recall', 'f1')
#     :return: The requested metric from the confusion matrix
#     """
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
#
#     # Initialize the classifier
#     classifiers = {
#         'logistic_regression': LogisticRegression(),
#         'svm': SVC(),
#         'random_forest': RandomForestClassifier(),
#         'naive_bayes': GaussianNB()
#     }
#     classifier = classifiers[classifier_type]
#
#     # Fit the classifier and predict
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#
#     # Compute confusion matrix and related metrics
#     cm = confusion_matrix(y_test, y_pred)
#     tp = cm[1, 1]  # True Positives
#     tn = cm[0, 0]  # True Negatives
#     fp = cm[0, 1]  # False Positives
#     fn = cm[1, 0]  # False Negatives
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred, average='binary'),
#         'recall': recall_score(y_test, y_pred, average='binary'),
#         'f1': f1_score(y_test, y_pred, average='binary'),
#         'tp': tp,  # True Positives
#         'tn': tn,  # True Negatives
#         'fp': fp,  # False Positives
#         'fn': fn  # False Negatives
#     }
#
#     if metric not in metrics:
#         raise ValueError("Unsupported metric. Choose from 'accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn'.")
#
#     # Return the requested metric
#     return metrics[metric]
#
#
# def split_data_on_attribute(data, features, target, attribute):
#     """
#     Splits the data into two groups based on the values (0 or 1) of a specified attribute and returns these subsets.
#
#     :param data: DataFrame containing the dataset
#     :param features: List of feature column names
#     :param target: Name of the target column
#     :param attribute: The attribute to split the data on, which should be binary (0 or 1)
#     :return: Two DataFrames, one for each value of the attribute
#     """
#     if attribute not in data.columns:
#         raise ValueError(f"The specified attribute '{attribute}' is not in the dataset.")
#
#     # Ensure the attribute contains only binary values
#     if sorted(data[attribute].unique()) != [0, 1]:
#         raise ValueError(f"The attribute '{attribute}' is not binary.")
#
#     # Split the data into two groups
#     group_0 = data[data[attribute] == 0]
#     group_1 = data[data[attribute] == 1]
#
#     return group_0, group_1
#


def calculate_metric_and_split_data(data, features, target, classifier_type='logistic_regression', metric='accuracy',
                                    attribute=None):
    """
    Train a classifier, predict on test data, calculate a specific metric from the confusion matrix,
    and optionally split the test data based on an attribute.

    :param data: DataFrame containing the dataset
    :param features: List of feature column names
    :param target: Name of the target column
    :param classifier_type: Type of classifier ('logistic_regression', 'svm', 'random_forest', 'naive_bayes')
    :param metric: Metric to calculate ('accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn')
    :param attribute: Optional attribute to split the test data on, which should be binary (0 or 1)
    :return: The requested metric from the confusion matrix and optionally two DataFrames, one for each value of the attribute
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the classifier
    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=500, solver='saga'),
        'svm': SVC(),
        'random_forest': RandomForestClassifier(),
        'naive_bayes': GaussianNB()
    }
    classifier = classifiers[classifier_type]

    # Fit the classifier and predict
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Compute confusion matrix and related metrics
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1, 1]  # True Positives
    tn = cm[0, 0]  # True Negatives
    fp = cm[0, 1]  # False Positives
    fn = cm[1, 0]  # False Negatives
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'tp': tp,  # True Positives
        'tn': tn,  # True Negatives
        'fp': fp,  # False Positives
        'fn': fn  # False Negatives
    }

    if metric not in metrics:
        raise ValueError(
            "Unsupported metric. Choose from 'accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn'.")

    result = metrics[metric]

    if attribute is not None:
        if attribute != target:
            raise ValueError(f"The attribute for splitting should be the same as the target.")

        # Add target to features
        X_test_df = pd.DataFrame(X_test, columns=features)
        X_test_df[target] = y_test.values

        # Split the test data into two groups
        group_0 = X_test_df[X_test_df[attribute] == 0]
        group_1 = X_test_df[X_test_df[attribute] == 1]

        return result, group_0, group_1

    return result
#
# # Example usage:
# data, features, target, _, _, _, _, _ = load_adult()  # Load data with a predefined function
# metric_value = calculate_confusion_matrix_metric(data, features, target, 'random_forest', 'accuracy')
# print(f"Classifier & Metric result: {metric_value:.2f}")
#
# # sample_data = load_sample_data()
# # print("Original Data:\n", sample_data)
# #
# # processed_data, numeric_cols, categorical_cols = preprocess_dataset(sample_data)
# # print("\nProcessed Data:\n", processed_data)
# # print("\nNumeric Columns:", numeric_cols)
# # print("Categorical Columns:", categorical_cols)
#
# # Example usage:
# data, features, target, _, _, _, _, _ = load_adult()  # Load data with a predefined function
# attribute = 'target'  # This should be replaced with the actual binary attribute name in your dataset
#
# try:
#     group_0, group_1 = split_data_on_attribute(data, features, target, attribute)
#     print("Group 0 Data:\n", group_0.head())
#     print("Group 1 Data:\n", group_1.head())
# except Exception as e:
#     print(str(e))

# Example usage
data, features, target, numeric_columns, categorical_columns, scaler, data_df_copy, _ = load_adult()
# features = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
#             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']
target = 'target'
attribute = 'target'

# Run the function
metric, group_0, group_1 = calculate_metric_and_split_data(data, features, target, classifier_type='logistic_regression', metric='accuracy', attribute=attribute)

# Print results
print(f"Accuracy Metric: {metric}")
print("Group 0 (first 5 rows):")
print(group_0.head())
print("Group 1 (first 5 rows):")
print(group_1.head())

# ------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Define the logistic_regression model globally
logistic_regression = LogisticRegression(max_iter=500, solver='saga')


def calculate_metrics_and_split_data(data, features, target, classifier_type='logistic_regression',
                                     metrics=['accuracy'], attribute=None):
    """
    Train a classifier, predict on test data, calculate specified metrics from the confusion matrix,
    and optionally split the test data based on an attribute.

    :param data: DataFrame containing the dataset
    :param features: List of feature column names
    :param target: Name of the target column
    :param classifier_type: Type of classifier ('logistic_regression', 'svm', 'random_forest', 'naive_bayes')
    :param metrics: List of metrics to calculate ('accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn')
    :param attribute: Optional attribute to split the test data on, which should be binary (0 or 1)
    :return: Dictionary of requested metrics and optionally two DataFrames, one for each value of the attribute,
             and the trained model instance
    """
    # Shuffle the dataset to ensure random distribution
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Debugging: Print the shape of the train and test sets
    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert back to DataFrame to maintain feature names
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)

    # Debugging: Print the first few rows of scaled data
    print("First few rows of X_train after scaling:")
    print(X_train.head())
    print("First few rows of X_test after scaling:")
    print(X_test.head())

    # Initialize the classifier
    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=500, solver='saga'),
        'svm': SVC(),
        'random_forest': RandomForestClassifier(),
        'naive_bayes': GaussianNB()
    }
    classifier = classifiers[classifier_type]

    # Fit the classifier and predict
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Debugging: Print first few predictions and actual values
    print("First few predictions:")
    print(y_pred[:10])
    print("First few actual values:")
    print(y_test[:10].values)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1, 1]  # True Positives
    tn = cm[0, 0]  # True Negatives
    fp = cm[0, 1]  # False Positives
    fn = cm[1, 0]  # False Negatives

    # Debugging: Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Calculate metrics
    metrics_dict = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    # Print accuracy
    print(f"Accuracy: {metrics_dict['accuracy']}")

    # Validate metrics
    results = {}
    for metric in metrics:
        if metric not in metrics_dict:
            raise ValueError(
                f"Unsupported metric '{metric}'. Choose from 'accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn'.")
        results[metric] = metrics_dict[metric]

    if attribute is not None:
        if attribute != target:
            raise ValueError(f"The attribute for splitting should be the same as the target.")

        # Add target to features
        X_test_df = X_test.copy()
        X_test_df[target] = y_test.values

        # Split the test data into two groups
        group_0 = X_test_df[X_test_df[attribute] == 0]
        group_1 = X_test_df[X_test_df[attribute] == 1]

        return results, group_0, group_1, classifier

    return results, classifier


data, features, target, numeric_columns, categorical_columns, scaler, data_df_copy, _ = load_compas()

target = 'race_0'
attribute = 'race_0'

# # Run the function with specified metrics and classifier type
# results, group_0, group_1 = calculate_metrics_and_split_data(data, features, target, classifier_type='logistic_regression', metrics=['accuracy', 'precision', 'recall'], attribute=attribute)

# # Print results
# print("Results with specified metrics and classifier type:")
# for metric, value in results.items():
#     print(f"{metric}: {value}")

# print("Group 0 (first 5 rows):")
# print(group_0.head())

# print("Group 1 (first 5 rows):")
# print(group_1.head())

results, group_0, group_1, trained_model = calculate_metrics_and_split_data(data, features, target, attribute=attribute)

# Print results
print("Results:")
for metric, value in results.items():
    print(f"{metric}: {value}")

print("Group 0 (first 5 rows):")
print(group_0.head())

print("Group 1 (first 5 rows):")
print(group_1.head())

import dice_ml

d = dice_ml.Data(dataframe=data,
                 continuous_features=["juv_fel_count", "juv_misd_count", "juv_other_count"],
                 outcome_name='two_year_recid')

m = dice_ml.Model(model=trained_model, backend="sklearn")
explainer = dice_ml.Dice(d, m, method="random")
input_datapoint = group_1[0:1]
group_1_cf = explainer.generate_counterfactuals(input_datapoint,
                                  total_CFs=20,
                                  desired_class="opposite",
)
# Visualize it
group_1_cf.visualize_as_dataframe(show_only_changes=True)