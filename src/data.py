import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    categorical_columns = ['Sex', 'ST_Slope', 'ChestPainType', 'RestingECG', 'ExerciseAngina']
    data = pd.get_dummies(data, categorical_columns, dtype='int')
    return data

def split_data(data):
    features = []
    for column in data.columns:
        if column != 'HeartDisease':
            features.append(column)

    X = data.loc[:, features]
    y = data.loc[:, 'HeartDisease']
    return train_test_split(X, y, train_size=.75, random_state=123, shuffle=True)

def scaled_data(data):
    scaler = StandardScaler()
    X_train, X_test, _, _ = split_data(data)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    data = load_data("data/heart.csv")
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, "HeartDisease")