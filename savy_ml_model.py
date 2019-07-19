import pandas as pd
import numpy as np

features_list=['Loan_amount','Loan_interest','Loan_duration',
               'Gender','Age','City',
               'Credit_history_rating', 'Loan_type',
               'Dafault_probability','Monthly_income','Monthly_exp_mortgage','Monthly_exp_loans',
               'Source_of_income', 'Work_duration', 'Education', 'Place_of_living',
               'Family_status', 'Type_of_asset', 'Family_monthly_income',
               'Monthly_expenditure_leasing', 'Monthly_expenditure_other','Last_debt']
               
# create features vector
X=df.loc[:,features_list]

# create response vector
y=df.loc[:,['LOAN_STATUS']]

# Partition into train and test
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)


# Decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score

dct=tree.DecisionTreeClassifier()
dct.fit(X_train,y_train)
# Predict on testing data
dct_predictions=dct.predict(X_test)
decision_tree_model_accuracy=accuracy_score(y_test,dct_predictions)

# Logistic regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs',multi_class='ovr',max_iter=10000)
logreg.fit(X_train,y_train)
# Predict on testing data
logreg_predictions=logreg.predict(X_test)
logistic_regression_mode_accuracy=accuracy_score(y_test,logreg_predictions)

# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 25))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
knn_predictions=knn.predict(X_test)
knn_model_accuracy=accuracy_score(y_test,knn_predictions)

# RFE (Recursive Feature Elimination)
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 10)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

sorted(zip(map(lambda x: round(x, 4),rfe.ranking_), features_list))


print(decision_tree_model_accuracy)
print(logistic_regression_mode_accuracy)
print(knn_model_accuracy)


