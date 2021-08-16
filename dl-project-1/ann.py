#import libraries
import numpy as np
import pandas as pd
#pip install tensorflow-cpu not tensorflow then test with command:
#python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

#tf.__version__ # should be 2.20

class Ann():

    def __init__(self, data):
        #ai parameters
        self.lhp = [6, 6, 1] # layer hyper parameters
        self.af = ['relu', 'sigmoid'] #activation functions (softmax must be used if using categories)
        self.op = 'adam' #optimiser = adam stocast grad decent (loss error on preditions)
        self.lf = 'binary_crossentropy' #loss function (for binary otherwise you use category_crossentropy)
        self.met = 'accuracy' #metrics
        self.bs = 32 #number of batches to take to train
        self.e = 100 # num of epochs

        #running the ai
        self.preprocessData(data)
        self.buildAnn()
        self.trainAnn()

    def preprocessData(self, file):
        #import dataset
        dataset = pd.read_csv(file)

        #matrix of features
        self.X = dataset.iloc[:, 3:-1].values # take all columns except last one. #we won't need row number or customer id, surname can be excluded
        self.y = dataset.iloc[:, -1].values # take last column of dataset

        #encoding categorical data (label encoding gender column)
        le = LabelEncoder()
        self.X[:, 2] = le.fit_transform(self.X[:,2])

        #one hot encoding of geography column
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') #1 means the 2nd column
        self.X = np.array(ct.fit_transform(self.X))

        #split data into training and testing data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size= 0.2, random_state = 0)

        #feature scaling (compulse for deep learning)
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)

    def buildAnn(self):
        # initialise ann
        self.ann = tf.keras.models.Sequential()
        #add input and first hidden layer
        self.ann.add(tf.keras.layers.Dense(units=self.lhp[0], activation=self.af[0]))
        #add second hidden layer
        self.ann.add(tf.keras.layers.Dense(units=self.lhp[1], activation=self.af[0]))
        #add output layer
        self.ann.add(tf.keras.layers.Dense(units=self.lhp[2], activation=self.af[1]))

    def trainAnn(self):
        #compile ann
        self.ann.compile(optimizer = self.op, loss = self.lf, metrics = self.met)

        #train ann on training set
        self.ann.fit(self.X_train, self.y_train, batch_size=self.bs, epochs=self.e)

    def predict(self, data):
        prediction = self.ann.predict(self.sc.transform([data])) > 0.5
        prediction = prediction[0][0]
        if prediction:
            print("this customer will leave the bank")
        else:
            print("this customer will leave the bank")

    def predictTest(self):
        self.y_pred = self.ann.predict(self.X_test)
        self.y_pred = (self.y_pred  > 0.5)
        #print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        print(accuracy_score(self.y_test, self.y_pred))

if __name__ == "__main__":
    #run artificial neural network
    ai = Ann('churn_data.csv')

    """
    Test ANN model by predicting if the customer with the following informations will leave the bank: 
        Geography: France
        Credit Score: 600
        Gender: Male
        Age: 40 years old
        Tenure: 3 years
        Balance: $ 60000
        Number of Products: 2
        Does this customer have a credit card? Yes
        Is this customer an Active Member: Yes
        Estimated Salary: $ 50000

    """
    #observation = [1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]
    #ai.predict(observation)
    ai.predictTest()
    