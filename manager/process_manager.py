import numpy as np

from hiper_params_search.searcher import ClassifierHipperParamsSearcher
from manager.history_manager import HistoryManager
from model_validator.result import ClassifierValidationResult
from model_validator.validator import BaseValidator


class ProcessManager:

    def __init__(self,
                 seed: int,
                 params_searcher: ClassifierHipperParamsSearcher,
                 validator: BaseValidator,
                 history_manager: HistoryManager,
                 save_history: bool = True,
                 history_index: int = None):
        self.params_searcher = params_searcher
        self.validator = validator
        self.history_manager = history_manager
        self.save_history = save_history
        self.history_index = history_index

        np.random.seed(seed)

    def process(self, number_interations: int = None):
        search_cv = self.__process_hiper_params_search(number_interations)
        validation_result = self.__process_validation(search_cv)
        self.__save_data_in_history(validation_result)
        self.__show_results(validation_result)

    def __process_hiper_params_search(self, number_interations: int = None):
        if self.history_index is None:
            return self.params_searcher.search_hipper_parameters(number_interations)
        else:
            return None

    def __process_validation(self, search_cv) -> ClassifierValidationResult:
        if search_cv is None:
            return self.history_manager.load_result_from_history(self.history_index)
        else:
            return self.validator.validate(search_cv)

    def __save_data_in_history(self, result: ClassifierValidationResult):
        if self.save_history and self.history_index is None:
            search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
            validation_time = self.validator.end_best_model_validation - self.validator.start_best_model_validation

            self.history_manager.save_result(result,
                                             search_time=self.__format_time(search_time),
                                             validation_time=self.__format_time(validation_time))

    def __show_results(self, result: ClassifierValidationResult):
        result.show_cross_val_metrics()

        search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
        validation_time = self.validator.end_best_model_validation - self.validator.start_best_model_validation

        print(f'Tempo da Busca            : {self.__format_time(search_time)}')
        print(f'Tempo da Validação        : {self.__format_time(validation_time)}')


    @staticmethod
    def __format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
