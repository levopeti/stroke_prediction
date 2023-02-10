import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_cm_keras(_model, _X_test, _y_test):
    predictions = _model.predict(_X_test)
    cm = confusion_matrix(_y_test.argmax(axis=1), predictions.argmax(axis=1), normalize="pred")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(_y_test.argmax(axis=1)))
    disp.plot()

