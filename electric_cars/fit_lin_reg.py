import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


def fit_lin_reg(features_list, target, data: pd.DataFrame) -> LinearRegression:
    # Encode the 'Zone Name' column
    # label_encoder = LabelEncoder()
    # data['Zone Name'] = label_encoder.fit_transform(data['Zone Name'])

    # Define the feature columns (excluding the target column "Traffic" and "Date" column)
    # feature_columns = data.columns.tolist()
    # feature_columns.remove('Traffic')
    feature_columns = features_list+['Date']

    # Define the target column
    target_column = target

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns], data[target_column], test_size=0.2, random_state=42
    )

    X_test_dates = X_test['Date']
    X_test = X_test.drop(columns=['Date'])
    X_train = X_train.drop(columns=['Date'])

    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the traffic on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression")
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R^2 Score: {r2}')

    # Plot outputs
    plt.scatter(X_test_dates, y_test, color="black")
    plt.plot(X_test_dates, y_pred, color="blue", linewidth=1)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    return model
