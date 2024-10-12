import time

from sklearn.model_selection import RandomizedSearchCV

from hiper_params_search.searcher import ClassifierHipperParamsSearcher


class ClassifierRandomHipperParamsSearcher(ClassifierHipperParamsSearcher):
    """
    Classe específica para buscar hiper parâmetros utilizando o metodo de pesquisa aleatória, onde um número específico
    de combinações será testada
    """

    def __init__(self,
                 data_x,
                 data_y,
                 params: dict[str, list],
                 estimator,
                 n_jobs: int = -1,
                 scoring: str ='accuracy',
                 log_level: int = 1):
        super().__init__(data_x, data_y, params, estimator, n_jobs, scoring, log_level)

    def search_hipper_parameters(self, number_iterations: int = None) -> RandomizedSearchCV:
        self.start_search_parameter_time = time.time()

        search = RandomizedSearchCV(estimator=self.estimator,
                                    param_distributions=self.params,
                                    cv=self.cv,
                                    n_jobs=self.n_jobs,
                                    verbose=self.log_level,
                                    n_iter=number_iterations,
                                    scoring=self.scoring)

        search.fit(X=self.data_x, y=self.data_y)

        self.end_search_parameter_time = time.time()

        return search
