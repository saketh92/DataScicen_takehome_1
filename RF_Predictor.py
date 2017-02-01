#Sai Saketh Nandagiri
#Creating a Random Classifier for given TestData
import numpy as np
import pandas as pd
#Application of RandomForestClassifier for modeling the data
from sklearn.ensemble import RandomForestClassifier

#Reading the csv document
InputData=pd.read_csv("C:\Users\saketh sai\Desktop\DataScience\DataScience\DataScience\Vish\InterviewTest.csv")

#Preparing Training and Testing Data
TrainData=InputData.ix[range(50000)]
TestData=InputData.ix[range(50000,len(InputData))]
print "Length of Training Data: "+str(len(TrainData))
print "Length of Testing Data: "+str(len(TestData))

#Unloading target column for both training and test data
TrainOutput=TrainData.pop("Output")
TestOutput=TestData.pop("Output")

#Sanity check
print "Q: Cols in Train and Test Data:"+str(TrainData.columns.equals(TestData.columns))

#Creating a RandomForest Classifier
#There are 1000 decision trees created to make the decision
rf=RandomForestClassifier(n_estimators=1000, criterion="entropy")

#Training the classifier
rf.fit(TrainData, TrainOutput)

#Testing the performance with TestData
Error=np.average(np.abs(TestOutput^rf.predict(TestData)))
print "Error in classification is: "+str(Error)

#Sample Test Data for representation
SampleTestData=TestData.ix[range(50000,50010)]
SampleTestOutput=rf.predict(SampleTestData)
print "Sample Test Output: "+str(SampleTestOutput)
print "Original Test Ouput: "+str(TestOutput.head(10))