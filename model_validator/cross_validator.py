import time

import numpy as np
from sklearn.model_selection import cross_val_score

from model_validator.result import ScikitLearnCrossValidationResult
from model_validator.validator import ScikitLearnBaseValidator


class CrossValidatorScikitLearn(ScikitLearnBaseValidator):
    """
    Classe que implementa a validação cruzada do modelo encontrado pela busca de hiper parâmetros de modelos do Scikit-Learn.
    """

    def __init__(self,
                 log_level: int = 0,
                 n_jobs: int = -1):
        super().__init__(log_level, n_jobs)

    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv,
                 scoring='accuracy') -> ScikitLearnCrossValidationResult:
        self.start_best_model_validation = time.time()

        scores = cross_val_score(estimator=searcher,
                                 X=data_x,
                                 y=data_y,
                                 cv=cv,
                                 n_jobs=self.n_jobs,
                                 verbose=self.log_level,
                                 scoring=scoring)

        self.end_best_model_validation = time.time()

        result = ScikitLearnCrossValidationResult(
            mean=np.mean(scores),
            standard_deviation=np.std(scores),
            median=np.median(scores),
            variance=np.var(scores),
            standard_error=np.std(scores) / np.sqrt(len(scores)),
            min_max_score=(round(float(np.min(scores)), 4), round(float(np.max(scores)), 4)),
            estimator=searcher.best_estimator_,
            scoring=scoring
        )

        return result
