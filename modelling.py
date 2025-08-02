from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def split_datos(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluar_modelo(modelo, X_test, y_test):
    pred = modelo.predict(X_test)
    print("Reporte de Clasificación:\n", classification_report(y_test, pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, pred))
