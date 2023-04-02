import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r'C:\Users\Krishi Thiruppathi\Desktop\heart.csv')
print('data')
data.describe()
data.target.value_counts()
data.describe().T
pd.crosstab(data.sex, data.target).plot(kind="bar",figsize=(10,5),color=['blue','red' ])

plt.xlabel('Sex (0 = Female, 1 = Male)') # X-Label

plt.xticks(rotation=0) # Get or set the current tick locations and labels of the x-axis.

plt.legend(["Haven't Disease", "Have Disease"]) # legend = Index

plt.ylabel('Frequency') # X-Label

plt.show()
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)

IQR = Q3 - Q1

((data < (Q1 - 1.5 * IQR)) | (data < (Q3 - 1.5 * IQR))).sum()
for feature in data:
    dataset=data.copy
    dataset(feature).hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.title(feature)
    plt.show()
for feature in data:
    dataset = data.copy()
    
if 0 in dataset[feature].unique():
    pass
else:
    dataset[feature] = np.log(dataset[feature])
    dataset.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()

outliers = []
def detect_outliers(values):
    Threshold = 3
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    for i in values:
        z_score = (i-mean_val)/std_val
        if np.abs(z_score) > Threshold:
            outliers.append(i)
    return outliers

out = detect_outliers(data['age'])
out
outliers = []
def detect_outliers(values):
    Threshold = 3
    mean_val = np.mean(values)
    std_val = np.std(values)
    for i in values:
        z_score = (i-mean_val)/std_val
        if np.abs(z_score) > Threshold:
            outliers.append(i)
    return outliers

out = detect_outliers(data['trestbps'])
out
outliers = []
def detect_outliers(values):
    Threshold = 3
    mean_val = np.mean(values)
    std_val = np.std(values)
    for i in values:
        z_score = (i-mean_val)/std_val
        if np.abs(z_score) > Threshold:
            outliers.append(i)
    return outliers
    
out = detect_outliers(data['chol'])
out
outliers = []
def detect_outliers(values):
    Threshold = 3
    mean_val = np.mean(values)
    std_val = np.std(values)
    for i in values:
        z_score = (i-mean_val)/std_val
        if np.abs(z_score) > Threshold:
            outliers.append(i)
    return outliers

out = detect_outliers(data['thalach'])
out
data.corr()['chol'].sort_values().plot(kind='bar')
plt.figure(figsize=(14,10))

sns.heatmap(data.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)

plt.ylim(15,0) # show us the exact number of values we want

plt.show()
from sklearn.model_selection import train_test_split

X = data.drop('target', axis=1).values
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(X_train,y_train)

lr_pred = lr_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,lr_pred))

import pickle 
pickle.dump(lr_model,open("mltmodel.pkl","wb"))
