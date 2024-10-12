import time

from sklearn.model_selection import GridSearchCV

from hiper_params_search.searcher import ClassifierHipperParamsSearcher


class ClassifierGridHipperParamsSearcher(ClassifierHipperParamsSearcher):
    """
    Classe específica para buscar hiper parâmetros utilizando o metodo de grid, onde todas as combinações de valores
    serão testadas.
    """

    def __init__(self, data_x,
                 data_y,
                 params: dict[str, list],
                 estimator,
                 n_jobs: int = -1,
                 scoring='accuracy',
                 log_level: int = 1):
        super().__init__(data_x, data_y, params, estimator, n_jobs, scoring, log_level)


    def search_hipper_parameters(self, number_iterations: int = None) -> GridSearchCV:
        search = GridSearchCV(estimator=self.estimator,
                              param_grid=self.params,
                              cv=self.cv,
                              n_jobs=self.n_jobs,
                              verbose=self.log_level,
                              scoring=self.scoring)

        self.start_search_parameter_time = time.time()

        search.fit(X=self.data_x, y=self.data_y)

        self.end_search_parameter_time = time.time()

        return search
