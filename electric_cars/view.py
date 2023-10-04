from matplotlib import pyplot as plt


def plot_column(table, column, zone_name):
    # Plot the population evolution for each zone
    # Initialize the figure
    mask = table['Zone Name'] == zone_name
    plt.figure(figsize=(14, 8))
    plt.plot(table.loc[mask, ['Date']], table.loc[mask, [column]], label=column)

    # Configure the plot
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()