import sys
import sklearn
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn import preprocessing
from scipy.stats import skew
import functools
import operator


def drop_outliers(data):
    data = data.drop(index=data[data.LotFrontage > 300].index)
    data = data.drop(index=data[(data.GrLivArea > 4000) & (data['SalePrice'] < 300000)].index)
    data = data.drop(index=data[data.TotalBsmtSF > 6000].index)
    return data

def fillna(data):
    data.PoolQC = data.PoolQC.fillna('No')
    data.MiscFeature = data.MiscFeature.fillna('No')
    data.Alley = data.Alley.fillna('No')
    data.Fence = data.Fence.fillna('No')
    data.FireplaceQu = data.FireplaceQu.fillna('No')
    data.Electrical = data.Electrical.fillna('SBrkr')
    for column in ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']:
        data[column] = data[column].fillna('No')
    data.GarageYrBlt = data.GarageYrBlt.fillna(0)
    for id in (lambda no_BsmtExposure_BsmtQaul: no_BsmtExposure_BsmtQaul[no_BsmtExposure_BsmtQaul.notnull()])(
        data[data.BsmtExposure.isnull()].BsmtQual
    ).index:
        data.at[id, 'BsmtExposure'] = 'No'
    for id in (lambda no_BsmtFinType2_BsmtQaul: no_BsmtFinType2_BsmtQaul[no_BsmtFinType2_BsmtQaul.notnull()])(
        data[data.BsmtFinType2.isnull()].BsmtQual
    ).index:
        data.at[id, 'BsmtFinType2'] = 'No'
    for column in ['BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']:
        data[column] = data[column].fillna('No')
    data.MasVnrArea = data.MasVnrArea.fillna(data[data.MasVnrType == 'BrkFace'].MasVnrArea.mean())
    data.MasVnrType = data.MasVnrType.fillna('BrkFace')
    data.LotFrontage = data.LotFrontage.fillna(data.LotFrontage.mean())
    return data

def fillallna(data):
    for column in data.select_dtypes(include = ["object"]).columns:
        if(data[column].isnull().sum() > 0):
            data[column] = data[column].fillna('No')
    for column in data.select_dtypes(exclude = ["object"]).columns:
        if(data[column].isnull().sum() > 0):
            data[column] = data[column].fillna(0)
    return data
    
def transform_features(data):
    data.MSSubClass = data.MSSubClass.replace({
                0: "No",
                20 : "1-St-1946-NAS", 30 : "1-St-1945-O", 40 : "1-St-WF-ATTIC-ALL-AGES",
                45 : "1-1/2-St-UNF-ALL-AGES", 50: "1-1/2-St-FALL-AGES",
                60 : "2-St-1946-N", 70 : "2-St-1944-O", 75 : "2-1/2-St-ALL-AGES", 
                80 : "Sp-OR-MULTI-L", 85 : "Sp-FOYER", 90 : "DUP-ALL-ST-AGES",
                120 : "1-St-PUD-1946-N", 150: "1-1/2 St-PUD-ALL-AGES",
                160 : "2-St-PUD-1946-N", 180 : "PUD-MUL-L-INCL-SPLIT-LEV/FOYER",
                190 : "2-FAMILY-CONV-ALL-St-AGES"})
    data.Street = data.Street.replace({"No": 0, "Grvl" : 1, "Pave" : 2})
    data.Alley = data.Alley.replace({"No": 0, "Grvl" : 1, "Pave" : 2})
    data.LotShape = data.LotShape.replace({"No": 0, "IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4})
    data.Utilities = data.Utilities.replace({"No": 0, "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4})
    data.ExterQual = data.ExterQual.replace({"No": 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5})
    data.ExterCond = data.ExterCond.replace({"No": 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5})
    data.BsmtQual = data.BsmtQual.replace({"No": 0, "No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5})
    data.BsmtCond = data.BsmtCond.replace({"No": 0, "No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5})
    data.BsmtExposure = data.BsmtExposure.replace({"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3})
    data.BsmtFinType1 = data.BsmtFinType1.replace({"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6})
    data.BsmtFinType2 = data.BsmtFinType2.replace({"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6})
    data.HeatingQC = data.HeatingQC.replace({"No": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5})
    data.KitchenQual = data.KitchenQual.replace({"No": 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5})
    data.Functional= data.Functional.replace({"No": 0, "Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8})
    data.FireplaceQu = data.FireplaceQu.replace({"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5})
    data.GarageFinish = data.GarageFinish.replace({"No": 0, "Unf": 1, "RFn": 2, "Fin": 3})
    data.GarageQual = data.GarageQual.replace({"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5})
    data.GarageCond = data.GarageCond.replace({"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5})
    data.PavedDrive = data.PavedDrive.replace({"N" : 0, "P" : 1, "Y" : 2})
    data.PoolQC = data.PoolQC.replace({"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4})
    data.MoSold = data.MoSold.replace({1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                       7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"})
    return data
    
def create_simple_features(data):
    data["SimplOverallQual"] = data.OverallQual.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    data["SimplOverallCond"] = data.OverallCond.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    data["SimplPoolQC"] = data.PoolQC.replace({0: 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2})
    data["SimplGarageCond"] = data.GarageCond.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplGarageQual"] = data.GarageQual.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplFireplaceQu"] = data.FireplaceQu.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplFireplaceQu"] = data.FireplaceQu.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplFunctional"] = data.Functional.replace({0: 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})
    data["SimplKitchenQual"] = data.KitchenQual.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplHeatingQC"] = data.HeatingQC.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplBsmtFinType1"] = data.BsmtFinType1.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    data["SimplBsmtFinType2"] = data.BsmtFinType2.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    data["SimplBsmtCond"] = data.BsmtCond.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplBsmtQual"] = data.BsmtQual.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplExterCond"] = data.ExterCond.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    data["SimplExterQual"] = data.ExterQual.replace({0: 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    return data

def create_genarilized_features(data):
    data["OverallGrade"] = data.OverallQual * data.OverallCond
    data["GarageGrade"] = data.GarageQual * data.GarageCond
    data["ExterGrade"] = data.ExterQual * data.ExterCond
    data["KitchenScore"] = data.KitchenAbvGr * data.KitchenQual
    data["FireplaceScore"] = data.Fireplaces * data.FireplaceQu
    data["GarageScore"] = data.GarageArea * data.GarageQual
    data["PoolScore"] = data.PoolArea * data.PoolQC
    data["SimplOverallGrade"] = data.SimplOverallQual * data.SimplOverallCond
    data["SimplExterGrade"] = data.SimplExterQual * data.SimplExterCond
    data["SimplPoolScore"] = data.PoolArea * data.SimplPoolQC
    data["SimplGarageScore"] = data.GarageArea * data.SimplGarageQual
    data["SimplFireplaceScore"] = data.Fireplaces * data.SimplFireplaceQu
    data["SimplKitchenScore"] = data.KitchenAbvGr * data.SimplKitchenQual
    data["TotalBath"] = data.BsmtFullBath + (0.5 * data.BsmtHalfBath) + data.FullBath + (0.5 * data.HalfBath)
    data["AllSF"] = data.GrLivArea + data.TotalBsmtSF
    data["AllFlrsSF"] = data["1stFlrSF"] + data["2ndFlrSF"]
    data["AllPorchSF"] = data.OpenPorchSF + data.EnclosedPorch + data["3SsnPorch"] + data.ScreenPorch
    data["HasMasVnr"] = data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, "Stone" : 1, "No" : 0})
    data["BoughtOffPlan"] = data.SaleCondition.replace({"No": 0, "Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    return data

def most_corelated_to_sale_price(data):
    return data.corr().nlargest(27, 'SalePrice')['SalePrice'].index

def polynomial_features(data, cols):
    for feature in cols[1:]:
        for d in [2, 3, 4, 5]:
            data[feature + '-d' + str(d)] = data[feature] ** d
    for features in [
                ['AllSF', 'OverallQual'],
                ['AllSF', 'ExterQual'],
                ['AllSF', 'KitchenQual'],
                ['AllSF', 'SimplOverallQual'],
                ['AllSF', 'GarageCars'],
                ['AllSF', 'TotalBath'],
                ['AllSF', 'TotalBsmtSF'],
                ['AllSF', 'BsmtQual'],
                ['AllSF', 'OverallGrade'],
                ['AllSF', 'ExterGrade'],
                ['AllSF', '1stFlrSF'],
                ['AllSF', 'AllFlrsSF'],
                ['AllSF', 'OverallQual', 'ExterQual'],
                ['AllSF', 'OverallQual', 'GarageCars'],
                ['AllSF', 'OverallQual', 'GarageCars', 'TotalBsmtSF'],
                ['AllSF', 'OverallQual', 'ExterQual', 'TotalBsmtSF'],
                ['AllSF', 'BsmtQual', 'GarageCars', 'TotalBsmtSF'],
                ['AllSF', 'BsmtQual', 'ExterQual', 'TotalBsmtSF'],
                ['AllSF', 'AllFlrsSF', 'GarageCars', 'TotalBsmtSF'],
                ['AllSF', 'OverallQual', 'ExterQual', 'KitchenQual'],
                ['AllSF', 'OverallQual', 'KitchenQual', 'GarageCars'],
                ['TotalBsmtSF', 'AllFlrsSF'],
                ['TotalBsmtSF', '1stFlrSF'],
                ['GarageCars', 'ExterQual', 'SimplOverallQual', 'BsmtQual', 'KitchenQual'],
                ['OverallQual', 'KitchenQual'],
                ['OverallQual', 'ExterQual'],
                ['OverallQual', 'KitchenQual'],
                ['OverallQual', 'BsmtQual'],
                ['OverallQual', 'ExterGrade'],
                ['OverallQual', 'OverallGrade'],
                ['OverallQual', 'GarageFinish'],
                ['OverallQual', 'ExterQual', 'KitchenQual'],
                ['OverallQual', 'ExterQual', 'BsmtQual'],
                ['OverallQual', 'ExterQual', 'OverallGrade'],
                ['OverallQual', 'ExterQual', 'GarageCars'],
                ['OverallQual', 'ExterQual', 'FireplaceScore'],
                ['OverallQual', 'ExterQual', 'KitchenQual', 'BsmtQual'],
                ['OverallQual', 'KitchenQual', 'BsmtQual', 'FireplaceScore'],
                ['ExterGrade', 'OverallGrade'],
                ['GarageCars', 'GarageScore', 'GarageFinish']]:
        data['-'.join(features)] = functools.reduce(operator.mul, list(map((lambda f: data[f]), features)), 1)           
    return data

def columns_split_by_kind_and_target(data):
    categorical_columns = data.select_dtypes(include = ["object"]).columns
    numerical_columns = data.select_dtypes(exclude = ["object"]).columns
    return numerical_columns.drop("SalePrice"), categorical_columns, data.SalePrice

def log_transform_skewed(data, numerical_columns, categorical_columns):
    numeric_data = data[numerical_columns]
    skewness = numeric_data.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    skewed_columns = skewness.index
    numeric_data[skewed_columns] = np.log1p(numeric_data[skewed_columns])
    return pd.concat([numeric_data, data[categorical_columns]], axis = 1)

def one_hot_encode_categorical(data, train, test, categorical_columns):
    one_hot_encoder = OneHotEncoder().fit(data[categorical_columns])
    return one_hot_encoder.transform(train[categorical_columns]), one_hot_encoder.transform(test[categorical_columns])

def stdScale_numeric(data, train, test, numerical_columns):
    #stdScaler = StandardScaler().fit(train[numerical_columns])
    #return stdScaler.transform(train[numerical_columns]), stdScaler.transform(test[numerical_columns])
    return train[numerical_columns], test[numerical_columns]

def missing(set_to_check):
    total_missing = set_to_check.isnull().sum().sort_values(ascending=False)
    total_missing = total_missing[total_missing > 0]
    percent_missing = (set_to_check.isnull().sum() / set_to_check.isnull().count()).sort_values(ascending=False)
    percent_missing = percent_missing[percent_missing > 0]
    missing_data = pd.concat([total_missing, percent_missing], axis=1, keys=['Total Missing', 'Percentage of Missing'])
    return missing_data

data = create_genarilized_features(
            create_simple_features(
                transform_features(
                    fillna(
                        drop_outliers(pd.read_csv('../data/train.csv', index_col=['Id']))
                    )
                )
            )
)

most_corelated_to_sale_price_columns = most_corelated_to_sale_price(data)

data = polynomial_features(data, most_corelated_to_sale_price_columns)

numerical_columns, categorical_columns, target = columns_split_by_kind_and_target(data)

data = log_transform_skewed(data, numerical_columns, categorical_columns)

#print(missing(data))

#x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=42)

#one_hot_x_train, one_hot_x_test = one_hot_encode_categorical(data, x_train, x_test, categorical_columns)

#scaled_x_train, scaled_x_test = stdScale_numeric(data, x_train, x_test, numerical_columns)

#new_x_train = pd.np.concatenate([one_hot_x_train.todense(), scaled_x_train], axis=1)
#new_x_test = pd.np.concatenate([one_hot_x_test.todense(), scaled_x_test], axis=1)

#test = create_genarilized_features(
#            create_simple_features(
#                transform_features(
#                    fillna(pd.read_csv('../data/test.csv', index_col=['Id']))
#               )
#            )
#)

print(missing(pd.read_csv('../data/test.csv', index_col=['Id'])))

#test = create_genarilized_features(
#           create_simple_features(
#               transform_features(
#                   fillallna(pd.read_csv('../data/test.csv', index_col=['Id']))
#               )
#           )
#)

#test = polynomial_features(test, most_corelated_to_sale_price_columns)

#test = log_transform_skewed(test, numerical_columns, categorical_columns)

#one_hot_x_train, one_hot_x_test = one_hot_encode_categorical(
#        pd.concat([data, test]),
#        data, test, categorical_columns)

#scaled_x_train, scaled_x_test = stdScale_numeric(data, data, test, numerical_columns)

#new_x_train = pd.np.concatenate([one_hot_x_train.todense(), scaled_x_train], axis=1)
#new_x_test = pd.np.concatenate([one_hot_x_test.todense(), scaled_x_test], axis=1)

#ridge = Ridge(alpha=0.6).fit(new_x_train, pd.np.log10(target))
#print(r2_score(10**ridge.predict(new_x_train), target))
#predicted = ridge.predict(new_x_test)

#print(10**ridge.predict(new_x_test))
