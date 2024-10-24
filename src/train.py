from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data import load_data, preprocess_data, scaled_data, split_data
import joblib

def train():
    data = load_data("data/heart_disease.csv")
    data = preprocess_data(data)
    X_train_scaled, X_test_scaled = scaled_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    # Multilayer Perceptron (Neural Network)
    model = MLPClassifier(
        hidden_layer_sizes=(10,),     # Number of neurons in the hidden layers
        activation='relu',            # Activation function for hidden layers
        solver='adam',                # Optimization algorithm
        max_iter=700,                 # Maximum number of iterations
        alpha=0.01,                   # L2 regularization term
        learning_rate="constant",      # Learning rate schedule
        random_state=123,             # Seed for reproducibility
        shuffle=True,                 # Make sure data is shuffled

    )

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Fit the model
    model.fit(X_train_scaled, y_train)   

    # save the model to disk
    filename = 'models/heart_disease_mlp.sav'
    joblib.dump(model, filename)

if __name__ == "__main__":
    train()