import time

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from model_validator.result import ClassifierBasicValidationResult
from model_validator.validator import BaseValidator


class BasicValidator(BaseValidator):

    def __init__(self,
                 data_x,
                 data_y,
                 cv,
                 log_level: int = 1,
                 scoring='accuracy',
                 test_size: float = 0.25):
        super().__init__(data_x, data_y, cv, log_level, scoring)

        self.test_size = test_size

    def validate(self, searcher) -> ClassifierBasicValidationResult:
        self.start_best_model_validation = time.time()

        x_train, x_test, y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=self.test_size)
        searcher.fit(x_train, y_train)

        predict_result = searcher.predict(x_test)

        accuracy = accuracy_score(y_test, predict_result)
        conf_matrix = confusion_matrix(y_test, predict_result)

        self.end_best_model_validation = time.time()

        return ClassifierBasicValidationResult(
            score=accuracy,
            estimator=searcher.best_estimator_,
            confusion_matrix=conf_matrix
        )
