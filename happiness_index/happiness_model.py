import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_digits
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# https://www.kaggle.com/unsdsn/world-happiness

#Load data, check data out!
raw_data = pd.read_csv('happiness_index/2015.csv')
# print(raw_data.head())
# print(raw_data.dtypes)
# print(raw_data.columns)

#define target, features, and features that need converting to numnbers
one_hot_features = []
features = ['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']
target = 'Happiness Score'

# print(features)
print(raw_data[features].describe())
# print(raw_data.columns)
# print(raw_data['Happiness Score'].value_counts())

#Test and visualise phase. Check out data and see what it looks like
# for f in ['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']:
#     plt.scatter(raw_data[f], raw_data[target])
#     plt.ylabel('Happiness Score')
#     plt.xlabel(f)
#     plt.title(f'{f} vs {target}')
#     plt.show()

# sns.scatterplot(x='Economy (GDP per Capita)', y='Happiness Score', data=raw_data, hue='Region')
# plt.legend(loc='best')
# plt.show()

#Heatmap showing correlation between variables
plt.figure(figsize=(11,11))
map_data = raw_data.drop(columns=['Happiness Rank', 'Standard Error', 'Country', 'Region'])
sns.heatmap(map_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


# make all one hot features strings, so they can then be encoded to numbers properly
for feature in one_hot_features:
    raw_data[feature] = raw_data[feature].astype(str)

for feature in features + [target]:
    if feature not in one_hot_features:
        raw_data[feature] = raw_data[feature].astype(float)


# perform one hot encoding
for feature in one_hot_features: 
    one_hot_enc = OneHotEncoder(categories='auto')   
    one_hot_encoded = one_hot_enc.fit_transform(raw_data[f'{feature}'].values.reshape(-1,1)).toarray()
    
    # convert back to data frame
    df_one_hot = pd.DataFrame(one_hot_encoded, columns = [f"{feature}_"+str(int(i)) for i in range(one_hot_encoded.shape[1])])
    raw_data = pd.concat([raw_data, df_one_hot], axis=1)
    features.extend([f"{feature}_"+str(int(i)) for i in range(one_hot_encoded.shape[1])])




#split data into training set and test set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

train, test = train_test_split(raw_data, test_size=0.20)

#scale features. First define each part - for training and test data sets
feature_train = train[features]
target_train = train[target]
feature_test = test[features]
target_test = test[target]

# learn scalers. Squishes feature data set into the same scale. Without this the RMSE doubles!!
feature_scaler = RobustScaler(quantile_range=(25, 75)).fit(feature_train)

# perform scaling
feature_train = feature_scaler.transform(feature_train)
feature_test = feature_scaler.transform(feature_test)

#Run Lasso model
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, classification_report, mean_squared_error

alpha = 0.01
lasso = Lasso(alpha=alpha)
lasso.fit(feature_train, target_train)
pred_train_lasso= lasso.predict(feature_train)

#here we're asking the model to predict the results of using the test data set. i.e. if we passed it next years survey numbers, it would predict the happiness score
# We'd need to pass in the features in the same order as we define above: print(lasso.predict([ [10, 10 ] ]))
pred_test_lasso = lasso.predict(feature_test)

#RMSE. Reslts are c.0.25, with the dependancy (happiness) ranging from 2.8-7.5. Therefore RMSE is small (good!)
# print("RMSE train set: ", np.sqrt(mean_squared_error(target_train,pred_train_lasso)))
# print("RMSE test set: ", np.sqrt(mean_squared_error(target_test,pred_test_lasso)))
# #R-squared values. values =0.95, 1 means the model explains the variability in happiness perfectly. So R-squared is high
# print("R-squared train set: ", r2_score(target_train, pred_train_lasso))
# print("R-squared test set: ", r2_score(target_test, pred_test_lasso))

train_score=lasso.score(feature_train,target_train)
test_score=lasso.score(feature_test,target_test)


print("Train score is: ", train_score)
print("Test score is: ", test_score)

print([x for x in zip(features, lasso.coef_) if x[1] > 0.0])



#testing it on 2016 data
alpha = 0.01
lasso2 = Lasso(alpha=alpha)
lasso2.fit(feature_train, target_train)

data_2016 = pd.read_csv('happiness_index/2016.csv')
features_2016 = data_2016[['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']]
target_2016 = data_2016['Happiness Score']

test_2016 = lasso2.predict(features_2016)
# score_2016=lasso2.score(features_2016)
# print(score_2016)
print("RMSE 2016 set: ", np.sqrt(mean_squared_error(target_2016,test_2016)))
print("R-squared test set: ", r2_score(target_2016, test_2016))