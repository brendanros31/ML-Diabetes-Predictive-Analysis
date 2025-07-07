from sklearn.preprocessing import StandardScaler



# Scaling data
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
