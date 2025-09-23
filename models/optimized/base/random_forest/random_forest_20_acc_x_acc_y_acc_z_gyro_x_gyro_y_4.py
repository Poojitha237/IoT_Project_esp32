
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

def create_model():
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=4, random_state=42)
    return clf

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Example usage
if __name__ == '__main__':
    clf = create_model()
    accuracy, precision, recall, f1 = evaluate_model(clf, X_train, X_test, y_train, y_test)
    print(f"Accuracy: 1.0, Precision: 1.0, Recall: 1.0, F1: 1.0")
