from pathlib import Path
import pandas as pd
from logging_config import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Numerical
# missing data - simple imputer (mean) on bmi
# scaling of numerical columns - StandardScalar
# Categorical
# one hot encoding of categorical columns - OHE

# num_pipeline = Pipeline(steps = [("imputer", SimpleImputer(num_cols) , ("scalar", StandardScalar(num_cols)))
# cat_pipeline = Pipeline(steps = [(imputer, SimpleImputer(cat_cols)) , ("ohe", OneHotEncoder(cat_cols)))

# preprocessor = ColumnTransformer([("num_pipeline", num_pipeline, num_cols), ("cat_pipeline", cat_pipeline, cat_columns)])
