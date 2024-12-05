# import semua tools diperlukan
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# baca file train
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


X = train_data[columnns_in_train_data]
cols_with_missing_values = [col for col in columnns_in_train_data if X[col].isnull().any()]

# penentuan train_X, val_X, train_y, dan val_y
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# encoding
copy_train_X = X.copy()
copy_val_X = val_X.copy()
copy_test_X = test_X.copy()
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)

copy_train_X[object_columns] = ordinal_encoder.fit_transform(X[object_columns])
copy_test_X[object_columns] = ordinal_encoder.transform(test_X[object_columns])


# define model
rf_model = RandomForestRegressor(random_state=1, max_leaf_nodes=500)
rf_model.fit(copy_train_X, y)

# predict values
test_preds = rf_model.predict(copy_test_X)
print(test_preds)

output = pd.DataFrame({'Id': test_data["Id"],
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

import rapihin
rapihin.rapihin("submission.csv")

