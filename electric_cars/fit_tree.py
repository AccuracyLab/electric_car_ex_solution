import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def fit_tree(features_list, target, data: pd.DataFrame) -> DecisionTreeRegressor:
    # One-hot Encoding for "Zone Name" column
    # data = pd.get_dummies(data, columns=['Zone Name'])

    # Define feature and target columns
    feature_columns = features_list

    # Define the target column
    target_column = target

    # Split the data
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree model (configured to overfit)
    model = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Output the metrics
    print("Tree Regression")
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R^2 Score: {r2}')

    # Visualizing the first tree in the forest
    from sklearn.tree import plot_tree
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True, feature_names=feature_columns, max_depth=3)
    plt.show()

    return model

