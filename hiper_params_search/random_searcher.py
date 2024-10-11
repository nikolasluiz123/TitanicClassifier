import time

from sklearn.model_selection import RandomizedSearchCV

from hiper_params_search.searcher import ClassifierHipperParamsSearcher


class ClassifierRandomHipperParamsSearcher(ClassifierHipperParamsSearcher):
    """
    Classe para realização de todos os processos que envolvem a busca de hiper parâmetros para classificadores do
    scikit-learn.
    """

    def __init__(self,
                 data_x,
                 data_y,
                 params: dict[str, list],
                 cv,
                 estimator,
                 n_jobs: int = -1,
                 scoring: str ='accuracy',
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

    def search_hipper_parameters(self, number_iterations: int = None) -> RandomizedSearchCV:
        """
            Função para realizar a pesquisa dos melhores parâmetros para o estimador RandomForestRegressor.

            :param number_iterations: Quantidade de vezes que o RandomizedSearchCV vai escolher os valores dos parâmetros
            fornecidos no parâmetro params

            :return: Retorna o objeto RandomizedSearchCV que poderá ser utilizado na função calculate_cross_val_score
            e obter as métricas.
        """

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
