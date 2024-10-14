import os

import numpy as np

from hiper_params_search.result_analyzer import SearchResultAnalyzer
from hiper_params_search.searcher import ClassifierHipperParamsSearcher
from manager.history_manager import HistoryManager
from model_validator.result import ClassifierValidationResult
from model_validator.validator import BaseValidator


class ProcessManager:
    """
    Classe responsável por centralizar os processos necessários para obter um modelo de machine learning utilizando
    o scikit-learn.
    """

    def __init__(self,
                 seed: int,
                 params_searcher: ClassifierHipperParamsSearcher,
                 validator: BaseValidator,
                 history_manager: HistoryManager,
                 result_analyzer: SearchResultAnalyzer,
                 save_history: bool = True,
                 history_index: int = None):
        self.params_searcher = params_searcher
        self.validator = validator
        self.history_manager = history_manager
        self.result_analyzer = result_analyzer
        self.save_history = save_history
        self.history_index = history_index

        np.random.seed(seed)

    def process(self, number_interations: int = None):
        """
        Função que realiza todos os processos para obter um modelo

        :param number_interations: Número de iterações, utilizado apenas quando a implementação do BaseSearchCV aceita
        """
        search_cv = self.__process_hiper_params_search(number_interations)
        validation_result = self.__process_validation(search_cv)
        self.__save_data_in_history(validation_result)
        self.__show_results(validation_result)
        self.__analyze_results(search_cv.cv_results_)

    def __process_hiper_params_search(self, number_interations: int = None):
        """
        Função para realizar a busca dos melhores parâmetros. Se for informado valor em history_index esse processo
        não precisa ser executado pois será pego do histórico.

        :param number_interations: Número de iterações, utilizado apenas quando a implementação do BaseSearchCV aceita
        """
        if self.history_index is None:
            return self.params_searcher.search_hipper_parameters(number_interations)
        else:
            return None

    def __process_validation(self, search_cv) -> ClassifierValidationResult:
        """
        Função que realiza a validação do modelo obtido. Se a instância do BaseSearchCV não for passada significa que
        não há necessidade de executar a validação e será retornado o objeto obtido do histórico.

        :param search_cv: Implementação de BaseSearchCV
        """
        if search_cv is None:
            return self.history_manager.load_result_from_history(self.history_index)
        else:
            return self.validator.validate(search_cv)

    def __save_data_in_history(self, result: ClassifierValidationResult):
        """
        Função para salvar os dados no histórico se save_history for True e a execução não tiver sido com base em um
        registro que já esta no histórico.

        :param result: Objeto com as métricas matemáticas calculadas
        """
        if self.save_history and self.history_index is None:
            search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
            validation_time = self.validator.end_best_model_validation - self.validator.start_best_model_validation

            self.history_manager.save_result(result,
                                             search_time=self.__format_time(search_time),
                                             validation_time=self.__format_time(validation_time))

    def __show_results(self, result: ClassifierValidationResult):
        """
        Função para exibir todas as métricas no console

        :param result: Objeto com as métricas matemáticas calculadas
        """

        result.show_cross_val_metrics()

        search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
        validation_time = self.validator.end_best_model_validation - self.validator.start_best_model_validation

        print(f'Tempo da Busca            : {self.__format_time(search_time)}')
        print(f'Tempo da Validação        : {self.__format_time(validation_time)}')


    def __analyze_results(self, cv_results):
        self.result_analyzer.show_boxplot_graphics_from_tested_params(cv_results)

        print()
        print('-' * 50)
        print('Piores Parâmetros: ')
        self.result_analyzer.show_bad_params(cv_results)

    @staticmethod
    def __format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
