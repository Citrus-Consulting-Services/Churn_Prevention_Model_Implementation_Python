###Importing the required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
from sklearn.externals import joblib 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

#Loading the data to datafrmae
Telco_data=pd.read_csv(‘TelcoCustomerChurn.csv') 
#See the first five rows of dataframe
Telco_data.head() 
#gives the number of rows and columns
Telco_data.shape 
#show all the columns in the dataset
Telco_data.columns  
Telco_data.isna().sum()   #gives the missing values
Telco_data.describe() #Give the statistics of the data

# Plotting target feature distribution
count = Telco_data['Churn'].value_counts(sort = True)
colors = ["grey","orange"] 
labels=['No','Yes']
#plotting pie chart
plt.pie(count,labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)
plt.title('Churn Percentage in data')
plt.show()

#plotting relation of target feature with independent attributes
sns.countplot(x ='TechSupport', hue = "Churn", data = Telco_data)
sns.countplot(x ='InternetService', hue = "Churn", data = Telco_data)

data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')  #Convert string to numeric

#Numerical features vs churn histogram
features = ['MonthlyCharges', 'tenure','TotalCharges']
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
Telco_data[Telco_data.Churn == 'No'][features].hist(bins=20, color="grey", alpha=0.9, ax=ax)
Telco_data[Telco_data.Churn == 'Yes'][features].hist(bins=20, color="orange", alpha=0.9, ax=ax)

#create a copy od dataset to clean the data
clean_df=Telco_data.copy()
#remove customer ID column from the dataset
Clean_df. drop('customerID',axis=1,inplace=True)
#to check the data datatypes
Clean_df.dtypes

#Converting string features to numerical features
Columns=clean_df.columns.values #to get all column names
for column in columns:
    if clean_df[column].dtype=='object':
        clean_df[column] = clean_df[column].astype('category')
        clean_df[column] = clean_df[column].cat.codes
    else:
        continue

#Creating Independent features and target variable
features=clean_df.drop('Churn',axis=1)
Y=clean_df['Churn']

#Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features

#Data Split
X_train, X_test, y_train, y_test = train_test_split(features,Y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#Intializing different algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Checking different model accuracies
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Box plot representation of accuracy of all the algorithms that we used
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results) #results of 10 iterations of each model
ax.set_xticklabels(names)
plt.show()

#Creating the model with the algorithm having higher accuracy
logreg = LogisticRegression() #Create the model
log_model=logreg.fit(X_train,y_train) #fit the training data to the model
print(log_model)

predicted_classes = log_model.predict(X_test) #predicts the test dataset class labels of each sample
predicted_prob=log_model.predict_proba(X_test) #predicts the probability of each sample in the test data
print(classification_report(y_test,predicted_classes)) #to get the accuracy metrics

#plotting ROC curve to choose the closest threshold
y_pred = []
for row in predicted_prob:
    y_pred.append(row[1])
y_pred = np.array(y_pred)
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
#plotting the curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#plotting confustion matrix to check false positives and false negatives
prediction = np.where(y_pred > 0.6, 1, 0) #using 0.6 as threshold to classify the label
f_mat = confusion_matrix(y_test, prediction)
df_cm = pd.DataFrame(f_mat, range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True, fmt='d')











