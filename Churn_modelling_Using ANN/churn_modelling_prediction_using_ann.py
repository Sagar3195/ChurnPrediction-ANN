"""##Churn modelling using ANN"""
####Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

###Loading dataset
df = pd.read_csv("/content/drive/My Drive/Churn_Model/Churn_Modelling.csv")

df.head()

df.shape

#lets checking missing values in dataset
df.isnull().sum()

#Now split data into independent variable and dependent variable
X = df.iloc[:,3:13]
y = df.iloc[:,-1]

X.shape, y.shape

#Now we convert categorical variable into numerical variables by using get dummies method
geography = pd.get_dummies(X['Geography'],drop_first= True)
gender = pd.get_dummies(X['Gender'], drop_first= True)

##Now we concate the both dataframe with df
X = pd.concat([X,geography, gender], axis = 1)
X.head()

X.drop(['Geography', 'Gender'], axis = 1, inplace = True)

X.head()

X.shape

#now feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#first we split dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train.shape[1]

import keras

#now we create ANN
from keras.models import Sequential #creating any neural networks
from keras.layers import Dense  #create hidden layers.
from keras.layers import LeakyReLU, PReLU, ELU  
from keras.layers import Dropout #create dropout layer & it used for regularization parameter.

#Now initializing ANN
classifier = Sequential()

"""1. Now create input layer and first hiddenn layer
,units parameter means we create 10 neurons in hidden layers.
2. Now we add dropout layer 
,dropout layer used for deep neural networks to reduce overfitting problem.
3. Also we dont have to create more layers bcoz it leads to overfitting.
"""

classifier.add(Dense(units = 10, kernel_initializer= 'he_normal', activation = 'relu', input_dim = x_train.shape[1]))
classifier.add(Dropout(0.3))

#now we adding seconde hidden layer.
classifier.add(Dense(units = 20, kernel_initializer='he_normal', activation= 'relu'))
classifier.add(Dropout(0.4))

#Now we adding third hidden layer.
classifier.add(Dense(units = 15, kernel_initializer= 'he_uniform', activation= 'relu'))
classifier.add(Dropout(0.2))

#Now adding output layer
classifier.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))

#now compiling ANN
#for binary classification we should use 'binay crossentropy'
#and for multiclass classification we use 'categorical crossentropy'
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#now fitting the ANN and training set
#validation_split for we can test data set separately for our test_data.
#batch_size use for computation power is lesss, more number of records dont have to load at one time
#and ram will be free and we can run this program freely.
model = classifier.fit(x_train, y_train, validation_split=0.33, batch_size= 10, epochs= 100)

"""#####After adding one more hidden layer and add another weight initlizer,

####After adding dropout layer we can see that accuracy is 84% and validation accuracy = 83%
"""

#now we predict model and check the accuracy of the model
y_predict = classifier.predict(x_test)
y_predict = (y_predict > 0.5)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_predict)
accuracy

"""#### We can see that accuracy for test data is 83%"""
