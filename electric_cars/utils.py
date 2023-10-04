import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .fit_lin_reg import fit_lin_reg
from .fit_tree import fit_tree
from .fit_xgboost import fit_xgboost

# label_encoder = LabelEncoder()

def load_data():
    # Load the dataset
    file_path = './descriptive_data.csv'
    data = pd.read_csv(file_path, sep=';', decimal='.')
    backup = pd.DataFrame()
    # Convert "Date" column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    zone_names = data['Zone Name'].unique()
    data['weekend'] = data['Date'].dt.weekday.apply(lambda x: 1 if x in [5, 6] else 0)

    # backup['Zone Name'] = data['Zone Name']
    # label_encoder = LabelEncoder()
    # data['Zone Name'] = label_encoder.fit_transform(data['Zone Name'])

    # Define the feature columns (excluding the target column "Traffic" and "Date" column)
    feature_columns = data.columns.tolist()
    feature_columns.remove('Traffic')

    # Extract feature from "Date" column
    future_area_data = data[data['Date'] > '2022-12-31']
    past_area_data = data[data['Date'] <= '2022-12-31']

    # Load Car Data
    # data['Zone Name'] = backup['Zone Name']
    # Hypothesis: 80% of charging needs are covered by home charging
    car_data = pd.read_csv('./car_data.csv', sep=';', decimal='.')
    car_data['Date'] = pd.to_datetime(car_data['Date'], format='%d/%m/%Y')

    # Load our Station Data
    owned_station_data = pd.read_csv('./our_stations_data.csv', sep=';', decimal='.')
    owned_station_data['Owned'] = True

    # Load Competitor Station Data
    station_data = pd.read_csv('./station_data.csv', sep=';', decimal='.')
    station_data['Owned'] = False

    return past_area_data, future_area_data, car_data, owned_station_data, station_data

def preview_table(table_name):
    display(table_name.head(10))
    return

def describe_column(table_name, column_name):
    display(table_name[column_name].describe())
    print('####################')
    print(table_name[column_name].value_counts())
    return

def fit_predicting_model(table_name, features_list, target, per_zone, model_type):
    # global label_encoder
    if per_zone:
        model = []
        for zone in ['Zone A', 'Zone B', 'Zone C']:
            table_ = table_name[(table_name['Zone Name'] == zone)]
            if model_type == 'linear_regression':
                model.append(fit_lin_reg(features_list, target, table_))
            elif model_type == 'decision_tree':
                model.append(fit_tree(features_list, target, table_))
            elif model_type == 'xgboost':
                model.append(fit_xgboost(features_list, target, table_))
        return model
    else:
        if "Zone_name" in features_list:
            table_name["_Zone_A"] = (table_name['Zone Name'] == 'Zone A').astype(float)
            table_name["_Zone_B"] = (table_name['Zone Name'] == 'Zone B').astype(float)
            table_name["_Zone_C"] = (table_name['Zone Name'] == 'Zone C').astype(float)
            features_list.remove("Zone_name")
            features_list.extend(["_Zone_A", "_Zone_B", "_Zone_C"])
        if model_type == 'linear_regression':
            return fit_lin_reg(features_list, target, table_name)
        elif model_type == 'decision_tree':
            return fit_tree(features_list, target, table_name)
        elif model_type == 'xgboost':
            return fit_xgboost(features_list, target, table_name)
        if "Zone_name" in features_list:
            table_name.drop(columns=["_Zone_A", "_Zone_B", "_Zone_C"], inplace=True)
    
def make_prediction(table_name, features_list, prediction_name, per_zone, model):
    if per_zone:
        table_name['A'] = model[0].predict(table_name[features_list])
        table_name['B'] = model[1].predict(table_name[features_list])
        table_name['C'] = model[2].predict(table_name[features_list])
        table_name[prediction_name] = (table_name['Zone Name'] == 'Zone A') * table_name['A'] + (table_name['Zone Name'] == 'Zone B') * table_name['B'] + (table_name['Zone Name'] == 'Zone C') * table_name['C']
        table_name.drop(columns=["A", "B", "C"], inplace=True)
    else:
        if "Zone_name" in features_list:
            table_name["_Zone_A"] = (table_name['Zone Name'] == 'Zone A').astype(float)
            table_name["_Zone_B"] = (table_name['Zone Name'] == 'Zone B').astype(float)
            table_name["_Zone_C"] = (table_name['Zone Name'] == 'Zone C').astype(float)
            features_list.remove("Zone_name")
            features_list.extend(["_Zone_A", "_Zone_B", "_Zone_C"])
        table_name[prediction_name] = model.predict(table_name[features_list])
        if "Zone_name" in features_list:
            table_name.drop(columns=["_Zone_A", "_Zone_B", "_Zone_C"], inplace=True)

    return table_name

def stack_tables(table_1, table_2, type):
    if type=='vertical':
        return pd.concat([table_1, table_2], axis=0)
    elif type=='horizontal':
        return pd.concat([table_1, table_2], axis=1)
    
def join_tables(table_1, table_2, join_key, join_type):
    return table_1.merge(table_2, on=join_key, how=join_type)

def apply_calculation_to_row(table_name, output_name, func, parameters_list):
    table_name[output_name] = table_name.apply(lambda row: func(*[row[p] for p in parameters_list]), axis=1)
    return table_name

def aggregate_sum(table, groupby_key, column_name):
    return table.groupby(groupby_key)[[column_name]].agg('sum').reset_index()
