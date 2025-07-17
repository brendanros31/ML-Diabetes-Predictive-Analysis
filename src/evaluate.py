import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import metrics



# Prediction Stats
def evaluate(model_obj: object, X_test, y_test: object):
        y_pred = model_obj.predict(X_test)   # Predictions
        
        print(model_obj)
        print(f'\n Confusion matrix: \n{metrics.confusion_matrix(y_test, y_pred)}\n')
        print(f'\nClassification report: \n{metrics.classification_report(y_test, y_pred)}')



# Model Visiual comparision
def visual_compare(model_name_list: list[str], model_obj_list: list[object], X_test, y_test):   
        
    list_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for model_obj in model_obj_list:
        y_pred = model_obj.predict(X_test)   # Prediction

        list_metrics["accuracy"].append(metrics.accuracy_score(y_test, y_pred))
        list_metrics["precision"].append(metrics.precision_score(y_test, y_pred, average='macro'))
        list_metrics["recall"].append(metrics.recall_score(y_test, y_pred, average='macro'))
        list_metrics["f1"].append(metrics.f1_score(y_test, y_pred, average='macro'))

    # Plot
    x = np.arange(len(model_name_list))
    width = 0.1

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - 1.5*width, list_metrics["accuracy"], width, label='Accuracy')
    ax.bar(x - 0.5*width, list_metrics["precision"], width, label='Precision')
    ax.bar(x + 0.5*width, list_metrics["recall"], width, label='Recall')
    ax.bar(x + 1.5*width, list_metrics["f1"], width, label='F1 Score')

    ax.set_xticks(x)

    ax.set_xticklabels(model_name_list)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

