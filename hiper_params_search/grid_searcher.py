import time

from sklearn.model_selection import GridSearchCV

from hiper_params_search.searcher import ClassifierHipperParamsSearcher


class ClassifierGridHipperParamsSearcher(ClassifierHipperParamsSearcher):

    def __init__(self, data_x,
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

        super().__init__(data_x, data_y, params, cv, estimator, n_jobs, scoring, log_level)
        self.data_x = data_x
        self.data_y = data_y
        self.params = params
        self.cv = cv
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.log_level = log_level


    def search_hipper_parameters(self, number_iterations: int = None) -> GridSearchCV:
        """
            Função para realizar a pesquisa dos melhores parâmetros para o estimador RandomForestRegressor.

            fornecidos no parâmetro params

            :return: Retorna o objeto RandomizedSearchCV que poderá ser utilizado na função calculate_cross_val_score
            e obter as métricas.
        """

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
