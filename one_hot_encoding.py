# import semua tools diperlukan
import pandas as pd
import numpy as np
import rapihin
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# baca file train
def predict_with_onehot_only():
    train_file = "D:\\Dowload-an Chrome\\house-prices-advanced-regression-techniques\\train.csv"
    train_data = pd.read_csv(train_file) # no missing values in file (checked)

    y = train_data["SalePrice"]

    columnns_in_train_data = list(set(train_data.columns) - set(["SalePrice", "Id"]))
    num_columns = [col for col in columnns_in_train_data if train_data[col].dtype != "object"]
    object_columns = [col for col in columnns_in_train_data if train_data[col].dtype == "object"]
    no_na_columns = [col for col in num_columns if not train_data[col].isnull().any()]

    # pembukaan file test
    test_file = "D:\\Dowload-an Chrome\\house-prices-advanced-regression-techniques\\test.csv"
    test_data = pd.read_csv(test_file)
    test_X = test_data[columnns_in_train_data]

    # penentuan fitur-fitur
    X = train_data[columnns_in_train_data]
    cols_with_missing_values = [col for col in columnns_in_train_data if X[col].isnull().any()]
    unique_val = train_data[object_columns].nunique(0)
    high_cardinality_columns = [col for col in object_columns if int(unique_val.loc[col]) > 10]
    low_cardinality_columns = list(set(object_columns)-set(high_cardinality_columns))

    # penentuan train_X, val_X, train_y, dan val_y
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # encoding
    copy_train_X = X.copy()
    copy_val_X = val_X.copy()
    copy_test_X = test_X.copy()
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_train_X = pd.DataFrame(one_hot_encoder.fit_transform(X[object_columns]))
    encoded_test_X = pd.DataFrame(one_hot_encoder.transform(test_X[object_columns]))

    encoded_train_X.index = X.index
    encoded_test_X.index = test_X.index

    fixed_train_X = pd.concat([encoded_train_X, X[num_columns]], axis=1)
    fixed_test_X = pd.concat([encoded_test_X, test_X[num_columns]], axis=1)

    fixed_train_X.columns = fixed_train_X.columns.astype(str)
    fixed_test_X.columns = fixed_test_X.columns.astype(str)

    # define model
    rf_model = RandomForestRegressor(random_state=1, max_leaf_nodes=500)
    rf_model.fit(fixed_train_X, y)

    # predict values
    test_preds = rf_model.predict(fixed_test_X)
    print(test_preds)

    output = pd.DataFrame({'Id': test_data["Id"],
                        'SalePrice': test_preds})
    output.to_csv('submission5.csv', index=False)

    rapihin.rapihin("submission5.csv")

def predict_with_onehot_and_dropna():
    train_file = "D:\\Dowload-an Chrome\\house-prices-advanced-regression-techniques\\train.csv"
    train_data = pd.read_csv(train_file) # no missing values in file (checked)

    y = train_data["SalePrice"]

    columnns_in_train_data = list(set(train_data.columns) - set(["SalePrice", "Id"]))
    num_columns = [col for col in columnns_in_train_data if train_data[col].dtype != "object"]
    object_columns = [col for col in columnns_in_train_data if train_data[col].dtype == "object"]
    no_na_columns = [col for col in num_columns if not train_data[col].isnull().any()]
    dropped_columns = ['GarageYrBlt', 'LotFrontage', 'MasVnrArea']

    # pembukaan file test
    test_file = "D:\\Dowload-an Chrome\\house-prices-advanced-regression-techniques\\test.csv"
    test_data = pd.read_csv(test_file)
    test_X = test_data[columnns_in_train_data]

    # penentuan fitur-fitur
    X = train_data[columnns_in_train_data]
    cols_with_missing_values = [col for col in columnns_in_train_data if X[col].isnull().any()]
    unique_val = train_data[object_columns].nunique(0)
    high_cardinality_columns = [col for col in object_columns if int(unique_val.loc[col]) > 10]
    low_cardinality_columns = list(set(object_columns)-set(high_cardinality_columns))

    # penentuan train_X, val_X, train_y, dan val_y
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    copy_train_X = X.copy()
    copy_test_X = test_X.copy()



    # encoding
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_train_X = pd.DataFrame(one_hot_encoder.fit_transform(X[low_cardinality_columns]))
    encoded_test_X = pd.DataFrame(one_hot_encoder.transform(test_X[low_cardinality_columns]))

    encoded_train_X.index = X.index
    encoded_test_X.index = test_X.index

    fixed_train_X = pd.concat([encoded_train_X, X[num_columns].drop(dropped_columns, axis=1)], axis=1)
    fixed_test_X = pd.concat([encoded_test_X, test_X[num_columns].drop(dropped_columns, axis=1)], axis=1)

    fixed_train_X.columns = fixed_train_X.columns.astype(str)
    fixed_test_X.columns = fixed_test_X.columns.astype(str)

    # define model
    rf_model = RandomForestRegressor(random_state=1, max_leaf_nodes=500)
    rf_model.fit(fixed_train_X, y)

    # predict values
    print(fixed_train_X.columns.tolist(), "\n")
    print(fixed_test_X.columns.tolist())
    test_preds = rf_model.predict(fixed_test_X)
    print(test_preds)

    output = pd.DataFrame({'Id': test_data["Id"],
                        'SalePrice': test_preds})
    output.to_csv('submission6.csv', index=False)

    rapihin.rapihin("submission6.csv")

predict_with_onehot_and_dropna()
# predict_with_onehot_only()