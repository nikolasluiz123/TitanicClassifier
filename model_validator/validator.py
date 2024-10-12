from abc import ABC, abstractmethod

from sklearn.model_selection import KFold

from model_validator.result import ClassifierCrossValidationResult


class BaseValidator(ABC):
    """
    Classe base que todos os validadores de modelo devem implementar
    """

    def __init__(self,
                 data_x,
                 data_y,
                 log_level: int = 1,
                 scoring='accuracy'):
        self.data_x = data_x
        self.data_y = data_y
        self.log_level = log_level
        self.scoring = scoring

        self.cv = KFold(n_splits=5, shuffle=True)
        self.start_best_model_validation = 0
        self.end_best_model_validation = 0

    @abstractmethod
    def validate(self, searcher) -> ClassifierCrossValidationResult:
        """
        Função para realizar a validação do modelo utilizando alguma estratégia.

        :return: Retorna um objeto CrossValScoreResult contendo as métricas matemáticas
        """
