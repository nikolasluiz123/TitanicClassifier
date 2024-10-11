from abc import ABC, abstractmethod


class ClassifierHipperParamsSearcher(ABC):

    def __init__(self,
                 data_x,
                 data_y,
                 params: dict[str, list],
                 cv,
                 estimator,
                 n_jobs: int = -1,
                 scoring='accuracy',
                 log_level: int = 1):
        """
            :param data_x: Features obtidas dos dados analisados
            :param data_y: Classes ou o resultado que deseja obter
            :param params: Hiper parâmetros que deseja testar
            :param cv: Estratégia de divisão dos grupos
            :param estimator Estimador que vai ser utilizado na busca
            :param n_jobs Thread do processador que serão utilizadas
        """

        self.params = params
        self.cv = cv
        self.data_x = data_x
        self.data_y = data_y
        self.n_jobs = n_jobs
        self.estimator = estimator
        self.scoring = scoring
        self.log_level = log_level

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

    @abstractmethod
    def search_hipper_parameters(self, number_iterations: int = None):
        ...
