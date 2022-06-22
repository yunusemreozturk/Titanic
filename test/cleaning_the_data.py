import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  # for missing data
from sklearn import preprocessing  # for Dummy Variable
from sklearn.model_selection import train_test_split  # to split data into train and test
from sklearn.preprocessing import StandardScaler  # for scaling

# here we import the data
train = pd.read_csv('../datasets/train.csv')
test = pd.read_csv('../datasets/test.csv')
test_survived = pd.read_csv('../datasets/gender_submission.csv')

# joining the test and train
all_test = pd.merge(test, test_survived, on='PassengerId')
all_data = pd.concat([train, all_test], axis=0)

# Age, Fare
imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
all_data[['Age', 'Fare']] = imputer_num.fit_transform(all_data[['Age', 'Fare']])

imputer_cab = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown')
all_data['Cabin'] = pd.DataFrame(imputer_cab.fit_transform(all_data['Cabin'].values.reshape(1309, 1)))

imputer_emb = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['Embarked'] = pd.DataFrame(imputer_emb.fit_transform(all_data['Embarked'].values.reshape(1309, 1)))

# edited the index
all_data = all_data.iloc[:, 1:]
all_data.reset_index(inplace=True)
all_data.drop(['index'], axis=1, inplace=True)

# Sex, Embarked, Cabin, Ticket
oe = preprocessing.OrdinalEncoder()

all_data['Sex'] = oe.fit_transform(all_data['Sex'].values.reshape(-1, 1))
all_data['Embarked'] = oe.fit_transform(all_data['Embarked'].values.reshape(-1, 1))
all_data['Cabin'] = oe.fit_transform(all_data['Cabin'].values.reshape(-1, 1))
all_data['Ticket'] = oe.fit_transform(all_data['Ticket'].values.reshape(-1, 1))

# Pclass, Embarked
ohe = preprocessing.OneHotEncoder(sparse=False)

pclass = pd.DataFrame(ohe.fit_transform(all_data['Pclass'].values.reshape(-1, 1)))
embarked = pd.DataFrame(ohe.fit_transform(all_data['Embarked'].values.reshape(-1, 1)))

embarked_df = pd.DataFrame(data=embarked.values, index=range(len(embarked)), columns=['C', 'Q', 'S'])
pclass_df = pd.DataFrame(data=pclass.values, index=range(len(pclass)), columns=['1st', '2nd', '3rd'])

# drop the Pclass, Embarked, Name
all_data = all_data.drop(['Pclass', 'Embarked'], axis=1)

result0 = pd.concat([embarked_df, pclass_df], axis=1)
result = pd.concat([result0, all_data], axis=1)

result.drop('Name', axis=1, inplace=True)
x = result.drop('Survived', axis=1)
y = result['Survived']

# splitting data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Nitel (Qualitative) (Kategorik)
# Survived: Sırasız (Nominal)
# Pclass: Sıralı (Ordinal)
# Name: Sırasız (Nominal)
# Sex: Sırasız (Nominal)
# Ticket: Sıralı (Ordinal)
# Cabin: Sıralı (Ordinal)
# Embarked: Sırasız (Nominal)

# Nicel (Quantitative) (Sayısal)
# PassengerId: Aralık (Interval)
# Age: Oransal (Ratio)
# SibSp: Aralık (Interval)
# Parch: Aralık (Interval)
# Fare: Oransal (Ratio)

