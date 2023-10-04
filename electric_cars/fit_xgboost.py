import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from matplotlib import pyplot as plt


def fit_xgboost(features_list, target, data: pd.DataFrame) -> xgb.XGBRegressor:
    # Define feature and target columns
    # Define feature and target columns
    feature_columns = features_list

    # Define the target column
    target_column = target

    # Split the data
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    model = xgb.XGBRegressor(objective ='reg:squarederror')
    model.fit(X_train, y_train)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Output the metrics
    print("XGBoost Regression")
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R^2 Score: {r2}')

    # (Optional) Plot feature importances
    xgb.plot_importance(model)
    plt.show()

    return model
