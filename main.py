import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = pd.read_csv('LoanApprovalPrediction.csv')

data.head(5)

#data preprocessing and visualization
obj = (data.dtypes == 'object')
print("Categorical variables: ", len(list(obj[obj].index)))

#As Loan_ID is completely unique
data = data.drop('Loan_ID', axis=1)
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18, 36))
index = 1

for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(6, 3, index)
    plt.xticks(rotation=45)
    sns.barplot(x = list(y.index), y = y)
    index += 1


#label encoder object to understand word labels
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

#to find out the number of columns with datatype == object
obj = (data.dtypes == 'object')
print("Categorical variables after encoding : ", len(list(obj[obj].index)))

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt=".2f", linewidths=2)

sns.catplot(x="Gender", y="LoanAmount", hue="Loan_Status", kind="bar", data=data)

#now find out if there is any missing value 
for col in data.columns:
    data[col] = data[col].fillna(data[col].median())

data.isna().sum()


#splitting the data into train and test
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
X.shape, Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


knn = KNeighborsClassifier(n_neighbors=7)
rfc = RandomForestClassifier(n_estimators=100)

svc = SVC()
lc = LogisticRegression()

for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy score of ", clf.__class__.__name__, "=",
          100 * metrics.accuracy_score(Y_test, Y_pred))