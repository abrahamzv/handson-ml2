
#%% SETUP
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import pandas as pd

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

#%% Create function to download data
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#%% Download
fetch_housing_data()

#%% Create function to load data
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#%% Load data
housing = load_housing_data()

#%%
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(10,8))
save_fig("attribute_histogram_plots")
plt.show()

#%% Simple train test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#%% Stratify based on income
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1,2,3,4,5])
housing['income_cat'].value_counts()
housing['income_cat'].hist()

#%% Stratified train test split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_train_set.drop('income_cat',axis=1, inplace=True)
strat_test_set.drop('income_cat', axis=1, inplace=True)
housing = strat_train_set.copy()



#%% Explore and visualize
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.8, figsize=(10,7), s=housing['population']/100, label='population', c='median_house_value', cmap='inferno')
plt.legend()


#%% Looking for correlations
corrmatrix = housing.corr()
corrmatrix['median_house_value'].sort_values(ascending=False)

#%%
from pandas.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,11));
housing.plot(kind='scatter', x='median_house_value', y='median_income', alpha=0.2, figsize=(8,8));

#%% Combine attributes
housing.head()
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing["population_per_household"] = housing['population']/housing['households']
housing.corr()['median_house_value'].sort_values(ascending=False)

#%% Separate labels and observations
housing = strat_train_set.drop(['median_house_value'], axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

#%% Impute missing observations with column medians
# Numerical attributes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing.drop('ocean_proximity', axis=1))
X = imputer.transform(housing.drop('ocean_proximity', axis=1))
housing_num = housing.drop('ocean_proximity', axis=1)
housing_tr = pd.DataFrame(X, index = housing.index, columns = housing_num.columns)

# Categorical attributes
housing_cat = housing[['ocean_proximity']]
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
cat_encoder.categories_

#%% Create a class to add attributes
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# %% codecell
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()

#%% Create pipelines for numerical and categorical data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
housing.shape
#%% Fit model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#%% Predict
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

some_predictions = np.round(lin_reg.predict(some_data_prepared), 2)

some_comparisons = pd.DataFrame({'labels': some_labels, 'predictions': some_predictions, 'diff': np.abs(some_labels-some_predictions)}, columns=['labels', 'predictions', 'diff'])
some_comparisons

#%%
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
rmse

#%%
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
rmse

#%%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)
rmse
def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', np.mean(scores))
    print('Std:', np.std(scores))

display_scores(rmse)

#%% Try random forests
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
np.sqrt(mse)

#Crossvalidated
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
rmse = np.sqrt(-scores)
display_scores(rmse)

#%% Save model
import joblib
cd experimentation/
joblib.dump(forest_reg, 'forest_reg.pkl')
loaded_mdl = joblib.load('forest_reg.pkl')
loaded_mdl.get_params()
help(joblib.dump)

#%%
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]}

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_.feature_importances_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

x = np.arange(housing_prepared.shape[1])
height = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

plt.bar(x, height, tick_label=attributes);
plt.xticks(x, attributes, rotation=90);

print(attributes)
cat_one_hot_attribs
full_pipeline.named_transformers_['cat'].categories_[0].shape
housing.head()
