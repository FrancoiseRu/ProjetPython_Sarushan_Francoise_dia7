# ProjetPython_Sarushan_Francoise_dia7


​ Dataset: https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset

The dataset represents the Facebook comment volume.
Shape of the data: We have 40 949 rows and 54 columns.​



Objective: Our goal is to predict how many comments a post is expected to receive in next H hours -> Column= 'Nb_comments_h_hour’

We did a preprocessing, cleanning and datavisualization.

Then, we test differents regression model with Grid Search CV :
-LinearRegression() : 0,5405
-SVR(linear): 0,4831
-SVR(rbf): 0,5554
-Ridge(): 0,5429
-ElasticNet(): 0,5420
-GradientBoostingRegressor(): 0.8754 

The best model is the Gradiant Boosting Regressor and the best gyperparameters is (learning_rate=0.03, max_depth=8, subsample=0.2)
