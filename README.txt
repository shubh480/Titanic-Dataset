

Topics covered in this Machine Learning Model
1. Introduction of data
2. Load the data
3. Fill the missing values
4. Feature engineering
5. Modeling
6. Prediction

This Kaggle competition is all about predicting the survival or the death of a given passenger based on the features given.
This machine learning model is built using scikit-learn and fastai libraries.RMS Titanic was a British passenger liner 
operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of April 15, 1912, after 
striking an iceberg during her maiden voyage from Southampton to New York City.The ship carried 1,317 passengers.

The Objective of this notebook is to give an idea how is the workflow in any predictive modeling problem. How do we check 
features, how do we add new features and some Machine Learning Concepts. I have tried to keep the notebook as basic as 
possible for better understanding.

Variable Notes
1. pclass: A proxy for socio-economic status (SES)
	1st = Upper
	2nd = Middle
	3rd = Lower

2. Age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

3. SibSp: The dataset defines family relations in this way...
	Sibling = brother, sister, stepbrother, stepsister
	Spouse = husband, wife (mistresses and fiancés were ignored)

4. parch: The dataset defines family relations in this way...
	Parent = mother, father
	Child = daughter, son, stepdaughter, stepson
	Some children travelled only with a nanny, therefore parch=0 for them.

Process of this project:
1. Input data is provided in folder named "titanic".
2. Use Titanic Kaggle.py file to run the code.
3. Used RandomForestClassifier model for prediction.
4. The result output of above code is stored in form of Titanic.csv

The code performed above gives a successful predicted results with an accuracy of 75% in the competition. 

Source: https://www.kaggle.com/vinothan/titanic-model-with-90-accuracy

