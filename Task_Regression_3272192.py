import pandas as pd
import requests
import zipfile
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

# Data loading and preprocessing functions
def load_and_preprocess_data():
    """
    Load and preprocess the bike sharing dataset. This dataset includes dates and other features related to the number of bike rentals.
    Returns the feature matrix X and the target variable y.
    """
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

    response = requests.get(data_url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('day.csv') as f:
            df = pd.read_csv(f)

    df['dteday'] =  pd.to_datetime(df['dteday'])
    df['day_of_month'] = df['dteday'].dt.day

    y = df['cnt']
    df = df.drop(['dteday', 'casual', 'registered', 'cnt'], axis=1)
    X = df

    return X, y

# Define the function for the random forest regression experiment
def random_forest_regression_experiment(n_estimators, min_samples_leaf, X, y):
    """
    Trains a RandomForestRegressor with the given parameters and performs 10-fold cross-validation.

    Parameters:
    - n_estimators: The number of trees in the forest.
    - min_samples_leaf: The minimum number of samples required for a leaf node.
    - X: Feature matrix (bike features).
    - y: Target vector (number of bike rentals).

    Returns:
    - The average negative mean squared error of the cross-validation.
    """

    regression = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=42)

    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(regression, X, y, scoring='neg_mean_squared_error', cv=kfolds)
    mse = scores.mean()
    # Return the computed mean negative mean squared error.
    return mse

# Function to find the best hyperparameter combination
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    """
    Find the best hyperparameter combination based on cross validation negative mean squared error.

    Parameters:
    - n_estimators_list: list of different values ​​for the number of trees.
    - min_samples_leaf_list: list of different values ​​for the minimum number of leaf samples.
    - X: feature matrix.
    - y: target vector.

    Returns:
    - The best hyperparameter combination and the corresponding negative mean squared error.
    """
    results = []
    best_result = {}

    for n in n_estimators_list:
        for m in min_samples_leaf_list:
            score = random_forest_regression_experiment(n_estimators=n, min_samples_leaf=m, X=X, y=y)
            results.append({'n_estimators': n,'min_samples_leaf': m,'mse': score})

    best_result = max(results, key=lambda x: x['mse'])

    # Return the best combination of hyperparameters and their corresponding smallest negative mean squared error
    return best_result # best_result is a dictionary


def manually_entered_best_params_and_mse():
    best_params = {'n_estimators': 1000, 'min_samples_leaf': 1}
    best_mse = -418641.1448
    return best_params, best_mse

if __name__ == "__main__":
    # Experiment with different values for n_estimators and min_samples_leaf
    # to find the best parameters setting among the following options:
    X, y = load_and_preprocess_data()
    n_estimators_list = [10, 50, 100, 1000] # Do not modify these parameter settings, experiment with them
    min_samples_leaf_list = [1, 10, 50, 100] # Do not modify these parameter settings, experiment with them

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y)
    print(f"Best Hyperparameter: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best negative mean square error: {best_params['mse']:.4f}")
