import time

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score

from hiper_params_search.history_manager import SearchParamsHistoryManager
from model_validation.classifier_validation import ClassifierCrossValScoreResult


class ClassifierGridHipperParamsSearch:

    def __init__(self,
                 data_x,
                 data_y,
                 params: dict[str, list],
                 history_dir: str,
                 history_file: str,
                 cv,
                 estimator,
                 seed: int = 1,
                 n_jobs: int = -1,
                 use_history: bool = True,
                 history_index: int = -1):
        """
            :param data_x: Features obtidas dos dados analisados
            :param data_y: Classes ou o resultado que deseja obter
            :param params: Hiper parâmetros que deseja testar
            :param history_dir Diretório onde serão salvos os dados de histórico da busca
            :param history_file Nome do arquivo JSON de histórico
            :param cv: Estratégia de divisão dos grupos
            :param estimator Estimador que vai ser utilizado na busca
            :param seed Seed para reprodutibilidade
            :param n_jobs Thread do processador que serão utilizadas
            :param use_history Flag que indica se deve recuperar um dado do histórico ou realizar o processamento completo
            :param history_index Índice utilizado para recuperar um dado do histórico caso use_history=True
        """

        self.params = params
        self.cv = cv
        self.data_x = data_x
        self.data_y = data_y
        self.n_jobs = n_jobs
        self.history_dir = history_dir
        self.history_file = history_file
        self.use_history = use_history
        self.history_index = history_index
        self.estimator = estimator

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

        self.start_best_model_cross_validation = 0
        self.end_best_model_cross_validation = 0

        self.history_manager = SearchParamsHistoryManager()

        np.random.seed(seed)

    def search_hipper_parameters(self) -> GridSearchCV:
        """
            Função para realizar a pesquisa dos melhores parâmetros para o estimador RandomForestRegressor.

            fornecidos no parâmetro params

            :return: Retorna o objeto RandomizedSearchCV que poderá ser utilizado na função calculate_cross_val_score
            e obter as métricas.
        """

        if not self.use_history or not self.history_manager.has_history():
            search = GridSearchCV(estimator=self.estimator,
                                  param_grid=self.params,
                                  cv=self.cv,
                                  n_jobs=self.n_jobs,
                                  verbose=1,
                                  scoring='accuracy')

            self.start_search_parameter_time = time.time()

            search.fit(X=self.data_x, y=self.data_y)

            self.end_search_parameter_time = time.time()

            return search

    def calculate_cross_val_score(self, searcher: GridSearchCV) -> ClassifierCrossValScoreResult:
        """
            Função para realizar a validação cruzada dos dados utilizando o resultado da busca de hiperparâmetros
            para validar o estimador encontrado.

            :return: Retorna um objeto CrossValScoreResult contendo as métricas matemáticas
        """

        if not self.use_history or not self.history_manager.has_history():
            self.start_best_model_cross_validation = time.time()

            scores = cross_val_score(estimator=searcher,
                                     X=self.data_x,
                                     y=self.data_y,
                                     cv=self.cv,
                                     n_jobs=self.n_jobs,
                                     verbose=1,
                                     scoring='accuracy')

            self.end_best_model_cross_validation = time.time()

            result = ClassifierCrossValScoreResult(
                mean=np.mean(scores),
                standard_deviation=np.std(scores),
                median=np.median(scores),
                variance=np.var(scores),
                standard_error=np.std(scores) / np.sqrt(len(scores)),
                min_max_score=(np.min(scores), np.max(scores)),
                iteration_number=self.__get_iteration_number(searcher)
            )

            search_time = self.end_search_parameter_time - self.start_search_parameter_time
            validation_time = self.end_best_model_cross_validation - self.start_best_model_cross_validation

            self.history_manager.save_result(result,
                                             search_time=self.__format_time(search_time),
                                             validation_time=self.__format_time(validation_time))

            return result

        else:
            return self.history_manager.load_result_from_history(index=self.history_index)

    def show_processing_time(self):
        """
        Função para exibir os tempos de execução de cada etapa do processo de busca de hiperparâmetros e validação cruzada
        no formato HH:MM:SS.
        """
        search_time = self.end_search_parameter_time - self.start_search_parameter_time
        validation_time = self.end_best_model_cross_validation - self.start_best_model_cross_validation

        print("Tempos de Execução")
        print("-" * 40)
        print(f"Tempo para busca de hiperparâmetros  : {self.__format_time(search_time)}")
        print(f"Tempo para validação cruzada          : {self.__format_time(validation_time)}")
        print("-" * 40)

    @staticmethod
    def __format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    def __get_iteration_number(self, searcher):
        iteration_number = 1

        for param, valores in searcher.param_grid.items():
            iteration_number *= len(valores)

        return iteration_number
