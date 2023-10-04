import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def correlation(data: pd.DataFrame):
    # Load the dataset
    outside_data = data.copy()
    for zone in data['Zone Name'].unique():
        data = outside_data[outside_data['Zone Name'] == zone].drop(columns=['Zone Name'])
        # Define the feature columns (excluding the target column "Traffic" and "Date" column)
        feature_columns = data.columns.tolist()
        feature_columns.remove('Traffic')
        feature_columns.remove('Date')
        target_column = 'Traffic'

        # Calculate the correlation matrix
        correlation_matrix = data[feature_columns + [target_column]].corr()

        # Set up the matplotlib figure
        plt.figure(figsize=(20, 15))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)

        # Set the title of the heatmap
        plt.title('Correlation Matrix')

        # Show the heatmap
        plt.show()


