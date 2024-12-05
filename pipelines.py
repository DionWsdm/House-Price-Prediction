import pandas as pd
import rapihin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# file-file
train_file = "D:\\Dowload-an Chrome\\house-prices-advanced-regression-techniques\\train.csv"
train_data = pd.read_csv(train_file) # no missing values in file (checked)

test_file = "D:\\Dowload-an Chrome\\house-prices-advanced-regression-techniques\\test.csv"
test_data = pd.read_csv(test_file)

# berbagai informasi mengenai kolom-kolom (feature)
columnns_in_train_data = list(set(train_data.columns) - set(["SalePrice", "Id"]))
num_columns = [col for col in columnns_in_train_data if train_data[col].dtype != "object"]
object_columns = [col for col in columnns_in_train_data if train_data[col].dtype == "object"]
no_na_columns = [col for col in num_columns if not train_data[col].isnull().any()]
cols_with_missing_values = [col for col in columnns_in_train_data if X[col].isnull().any()]
unique_val = train_data[object_columns].nunique(0)
high_cardinality_columns = [col for col in object_columns if int(unique_val.loc[col]) > 10]
low_cardinality_columns = list(set(object_columns)-set(high_cardinality_columns))

# X, y, dan test_X
X = train_data[columnns_in_train_data]
y = train_data["SalePrice"]
test_X = test_data[columnns_in_train_data]
copy_train_X = X.copy()
copy_test_X = test_X.copy()

# transformers
numerical_transforer = SimpleImputer(strategy="median")
object_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

# gabungkan numerical dan object transformers
preprocessor = ColumnTransformer(transformers=[("num", numerical_transforer, num_columns), ("cat", object_transformer, object_columns)])

# model
model = RandomForestRegressor(max_leaf_nodes=500, random_state=1)

# pipeline
mypipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
mypipeline.fit(X, y)

# predictions
test_preds = mypipeline.predict(test_X)

# output
output = pd.DataFrame({'Id': test_data["Id"], 'SalePrice': test_preds})
output.to_csv('submission7.csv', index=False)
rapihin.rapihin("submission7.csv")
