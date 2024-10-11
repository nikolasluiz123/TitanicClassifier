from abc import ABC, abstractmethod

from model_validator.result import ClassifierCrossValidationResult


class BaseValidator(ABC):

    def __init__(self,
                 data_x,
                 data_y,
                 cv,
                 log_level: int = 1,
                 scoring='accuracy'):
        self.data_x = data_x
        self.data_y = data_y
        self.cv = cv
        self.log_level = log_level
        self.scoring = scoring
        self.start_best_model_validation = 0
        self.end_best_model_validation = 0

    @abstractmethod
    def validate(self, searcher) -> ClassifierCrossValidationResult:
        ...
