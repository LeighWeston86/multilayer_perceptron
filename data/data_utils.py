import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'pima-indians-diabetes.data.txt')
    df = pd.read_csv(data_path, header = None)
    return df

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    return X_train_transformed, X_test_transformed

def get_data():
    df = load_data()
    X = df.iloc[:,:-1].as_matrix()
    y = df.iloc[:, -1].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = normalize_features(X_train, X_test)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    get_data()

