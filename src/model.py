from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def build_model(model_type, params=None):
    if model_type == 'GaussianNB':
        return GaussianNB(**(params or {}))
    
    elif model_type == 'LogisticRegression':
        return LogisticRegression(**(params or {}))
    
    elif model_type == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**(params or {}))
    
    elif model_type == 'RandomForestClassifier':
        return RandomForestClassifier(**(params or {}))



def train_model(model_obj, X_train, y_train):
    return model_obj.fit(X_train, y_train)