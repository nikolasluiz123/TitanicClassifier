from abc import ABC, abstractmethod

from sklearn.model_selection import KFold


class ClassifierHipperParamsSearcher(ABC):
    """
    Classe base de pesquisa de hiper parâmetros de algoritimos de classificação do scikit-learn.
    """

    def __init__(self,
                 data_x,
                 data_y,
                 params: dict[str, list],
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
            :param scoring O que o searcher vai utilizar para definir o melhor modelo
            :param log_level Nível dos logs exibidos no processo de busca, varia em 1, 2 e 3
        """

        self.params = params
        self.data_x = data_x
        self.data_y = data_y
        self.n_jobs = n_jobs
        self.estimator = estimator
        self.scoring = scoring
        self.log_level = log_level

        self.cv = KFold(n_splits=5, shuffle=True)
        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

    @abstractmethod
    def search_hipper_parameters(self, number_iterations: int = None):
        """
        Função que deve realizar a busca dos hiper parâmetros.

        :param number_iterations: Número de iterações que o searcher vai realizar. Seu valor é opcional pois há
        implementações que não aceitam um limite de iterações.

        :return: Retorna uma instância de algum BaseSearchCV depois de fazer o fit
        """
        ...
