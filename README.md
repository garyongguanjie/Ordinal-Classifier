# Ordinal Classifier

Compatible with sklearn

Implemented Implemented method described in this research paper https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf

Allows us to classify classes with an order to it.

For example cold,warm,hot.

Using traditional classification methods would cause us to lose ordinal data i.e. warm is between cold and hot

While using regression methods would not be a good fit as all values of a particular class are mapped to one value.

This classifier breaks down the problem to k-1 binary classification and then calculate the probability of it in each class. The highest probability is then the predicted class.

API is similar to sklearn

you must pass in an sklearn classifier that has predict_proba implemented

Y-values must start from 0 and increment by 1 for each class to indicate the order of the classes
```
from sklearn.linear_model import LogisticRegression
O = OrdClass(classifier=LogisticRegression)
O.fit(X,y)
O.predict(x_test)
```