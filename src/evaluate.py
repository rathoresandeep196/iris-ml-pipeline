import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_model():
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    clf = joblib.load("models/iris_model.pkl")
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    evaluate_model()
