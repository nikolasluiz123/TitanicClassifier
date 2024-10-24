from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from model_validator.result import ScikitLearnCrossValidationResult


class ScikitLearnBaseValidator(ABC):
    """
    Classe base para implementar validadores de estimadores do Scikit-Learn.
    """

    def __init__(self,
                 log_level: int = 1,
                 n_jobs: int = -1):
        self.log_level = log_level
        self.n_jobs = n_jobs

        self.start_best_model_validation = 0
        self.end_best_model_validation = 0

    @abstractmethod
    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv,
                 scoring='accuracy') -> ScikitLearnCrossValidationResult:
        """
        Função para realizar a validação do modelo utilizando alguma estratégia.

        :return: Retorna um objeto CrossValScoreResult contendo as métricas matemáticas
        """


class ClassifierFinalValidator:
    """
    Validador que pode ser utilizado por fora do ProcessManager de forma opcional para vizualizar algumas validações
    específicas de classificação.
    """

    def __init__(self, estimator, data_x, data_y):
        self.estimator = estimator
        self.data_x = data_x
        self.data_y = data_y

    def validate(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=42)

        self.estimator.fit(x_train, y_train)
        y_pred = self.estimator.predict(x_test)

        self.__show_classification_report(y_test, y_pred)
        self.__show_confusion_matrix(y_test, y_pred)

    def __show_confusion_matrix(self, y_test, y_pred):
        matrix = confusion_matrix(y_test, y_pred)

        classes = np.unique(np.concatenate([y_test, y_pred]))
        class_labels = [f"Classe {cls}" for cls in classes]

        df_cm = pd.DataFrame(matrix, index=class_labels, columns=class_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Matriz de Confusão")
        plt.ylabel("Classe Real")
        plt.xlabel("Classe Prevista")
        plt.show()

    def __show_classification_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        print()
        print('Relatório de Classificação:\n')
        print(tabulate(df_report, headers='keys', tablefmt="fancy_grid"))
