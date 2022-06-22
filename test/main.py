import cleaning_the_data as ctd
from sklearn.linear_model import LogisticRegression  # Linear Regression
from sklearn.metrics import confusion_matrix  # confusion matrix

x_train = ctd.x_train
x_test = ctd.x_test
y_train = ctd.y_train
y_test = ctd.y_test

X_train = ctd.X_train
X_test = ctd.X_test

# Linear Regression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('Logistic Regression')
print(cm)

# K-NN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('K-NN')
print(cm)

# SVC
from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('SVC - kernel=rbf')
print(cm)

# Naive Bayes - Gassian
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('Navie Bayes - Gassian')
print(cm)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('Decision Tree')
print(cm)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')

rfc.fit(X_train, y_train.ravel())

y_pred = rfc.predict(X_test)
y_proba = rfc.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)

print('Random Forest')
print(cm)
print(y_proba)

# Xgboost
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_pred, y_test)

print('XGboost')
print(cm)
