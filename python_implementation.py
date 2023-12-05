#Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#If the file already exists in the project repo uncomment the below line
#data = pd.read_csv("diabetes1.csv")

## Else load the dataset 
data=pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')

# See the first few lines of data 
data.head()

# Showing the statistical info of data
data.describe()

# Following shows the macro picture
data.info()

sns.countplot(x='Pregnancies',data=data)
# Maximum patients have conceived  1 and 0 times.

plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1

for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.histplot(data[column])
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Count',fontsize=20)
    plotnumber+=1
plt.tight_layout()
    
## Bivariate Analysis

## Analyzing how preganancies impact the patient with diabetes.
sns.countplot(x='Pregnancies',hue='Outcome',data=data)
plt.show()


## Aanlyzing the relationship between diabetes and Glucose
sns.histplot(x='Glucose',hue='Outcome',data=data)

## Analyze Glucose with blood pressure
sns.relplot(x='Glucose',y='BloodPressure',hue='Outcome',data=data)
plt.show()

## Analyze Glucose with SkinThickness
sns.relplot(x='Glucose',y='SkinThickness',hue='Outcome',data=data)
plt.show()

## Analyze relationship between BloodPressure and Outcome

sns.histplot(x='BloodPressure',hue='Outcome',data=data)

# Analyze BP with SkinThickness

sns.relplot(x='BloodPressure',y='SkinThickness',hue='Outcome',data=data)
plt.show()

# Analyze BP with Insulin

sns.relplot(x='BloodPressure',y='Insulin',col='Outcome',data=data)
plt.show()

# Analyzing Insulin with target
sns.histplot(x='Insulin',hue='Outcome',data=data)

#Step 1 Handling the missing values
data.isnull().sum()

# Step 2 Handling the corrupted data.
data.describe()

## In 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI' certain datapoints are zero.

data.loc[data['Glucose']==0]

data.Glucose.replace(0,np.mean(data.Glucose),inplace=True)

#dataframe.column.replace('Value to be replaced','By what value')

data.loc[data['Glucose']==0]
data.BloodPressure.replace(0,np.mean(data.BloodPressure),inplace=True)
data.SkinThickness.replace(0,np.median(data.SkinThickness),inplace=True)
data.Insulin.replace(0,np.median(data.Insulin),inplace=True)
data.BMI.replace(0,np.mean(data.BMI),inplace=True)

data.describe()

# Step 3:-Numerical representation of string data
import warnings
warnings.filterwarnings("ignore")

## Step 4:-Checking the outliers
plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1

for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.boxplot(data[column])
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Count',fontsize=20)
    plotnumber+=1
plt.tight_layout()

# Step 5:-Scaling the data

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
dl=['Pregnancies','Outcome']
data1=sc.fit_transform(data.drop(dl,axis=1))

con_data=data[['Pregnancies','Outcome']]

data1

data.columns

type(data1)
data2=pd.DataFrame(data1,columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])

final_df=pd.concat([data2,con_data],axis=1)

final_df

# No redundant fetaures
# We will check correlation
sns.heatmap(data2.corr(),annot=True)

# So no correlation hence no features should be 

# Step 1 Creating independent and dependent variable.

X=final_df.iloc[:,:-1]
y=final_df.Outcome

# Step 2 Creating training and testing data.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=45)

y_test.shape

# Step 3 Model creation
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()

clf.fit(X_train,y_train)  ## training

# Step 4 Prediction
y_pred=clf.predict(X_test)

y_pred
y_pred_prob=clf.predict_proba(X_test)
y_pred_prob

data.Outcome.value_counts()

# Evaluation of model
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score, precision_score,classification_report,f1_score
cm=confusion_matrix(y_test,y_pred)
print(cm)

pd.crosstab(y_test,y_pred)

Acc = accuracy_score(y_test,y_pred)
Acc

recall=recall_score(y_test,y_pred)
recall

precision=precision_score(y_test,y_pred)
precision

f1score=f1_score(y_test,y_pred)
f1score

cr=classification_report(y_test,y_pred)
print(cr)

## Testing the model
y_test.value_counts()
