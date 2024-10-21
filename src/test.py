from data import load_data, preprocess_data, split_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from data import scaled_data
import joblib
from prettytable import PrettyTable

def evaluate():
    # Load the model
    model = joblib.load("models/heart_disease_mlp.sav")

    # Load and preprocess the data
    data = load_data("data/heart_disease.csv")
    data = preprocess_data(data)
    X_train_scaled, X_test_scaled = scaled_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Evaluate the model
    nn_train_predictions = model.predict(X_train_scaled)
    nn_train_accuracy = accuracy_score(y_train, nn_train_predictions)

    nn_test_predictions = model.predict(X_test_scaled)
    nn_test_accuracy = accuracy_score(y_test, nn_test_predictions)
    
    print(classification_report(y_test, nn_test_predictions))
    
    nn_cv_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    
    # Initialize PrettyTable; for better visualization
    result_table = PrettyTable()
    result_table.field_names = ["Model", "Training Accuracy", "Testing Accuracy", "Cross-Validation Accuracy"]

    # Multilayer Perceptron result in the table
    result_table.add_row(["Multilayer Perceptron", f"{nn_train_accuracy:.4f}", f"{nn_test_accuracy:.4f}", f"{nn_cv_accuracy:.4f}"])

    # Print the table
    print(result_table)

if __name__ == "__main__":
    evaluate()
