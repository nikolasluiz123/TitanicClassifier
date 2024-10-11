import json
import os
import pickle

from model_validation.classifier_validation import ClassifierCrossValScoreResult


class SearchParamsHistoryManager:
    """
    Classe responsável por armazenar os dados históricos das buscas de hiper parâmetros dos modelos. Isso pode evitar
    um reprocessamento quando desejar apenas exibir novamente no console um resultado obtido em uma das tentativas.

    Os dados referentes ao desempenho do modelo são salvos no formato de JSON, dentro de uma lista, onde poderão ser
    recuperados através do seu índice. Além disso, o próprio modelo é salvo para que possa ser utilizado com dados diferentes
    e possa ser verificado o seu comportamento.
    """

    def __init__(self):
        self.output_directory = 'history'
        self.params_file_name = 'tested_params'

    def save_result(self, classifier_result: ClassifierCrossValScoreResult, search_time: str, validation_time: str):
        """
        Função utilizada para salvar os resultados obtidos da busca de hiper parâmetros.

        :param classifier_result: Objeto com os dados do resultado da busca
        :param search_time: Tempo que demorou para realizar a busca dos melhores parâmetros
        :param validation_time: Tempo que demorou para realizar a validação cruzada com o melhor modelo
        """
        dictionary = {
            'mean': classifier_result.mean,
            'standard_deviation': classifier_result.standard_deviation,
            'median': classifier_result.median,
            'variance': classifier_result.variance,
            'standard_error': classifier_result.standard_error,
            'min_max_score': classifier_result.min_max_score,
            'estimator_params': classifier_result.estimator.get_params(),
            'number_iterations': classifier_result.iteration_number,
            'search_time': search_time,
            'validation_time': validation_time
        }

        self.__create_output_dir()
        self.__save_dictionary_in_json(dictionary)
        self.__save_model(classifier_result.estimator)

    def __create_output_dir(self):
        """
        Função para criar o diretório de histório caso não exista. É nesse diretório que o arquivo JSON e os modelos
        ficarão.
        """
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def __save_dictionary_in_json(self, dictionary):
        """
        Função utilizada para adicionar o dicionário com os valores resultantes da busca dentro da lista do JSON

        :param dictionary: Dicionário com os dados
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                data = json.load(file)
        else:
            data = []

        data.append(dictionary)

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

    def has_history(self) -> bool:
        """
        Retorna se há ao menos um registro dentro do arquivo de histórico
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return False

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data) > 0

    def load_result_from_history(self, index: int = -1) -> ClassifierCrossValScoreResult:
        """
        Carrega o objeto ClassifierCrossValScoreResult com os dados do histórico.

        :param index Índice utilizado para recuperar da lista de parâmetros o resultado que deseja visualizar novamente
        """
        result_dict = self.__get_dictionary_from_json(index)

        return ClassifierCrossValScoreResult(
            mean=result_dict['mean'],
            standard_deviation=result_dict['standard_deviation'],
            median=result_dict['median'],
            variance=result_dict['variance'],
            standard_error=result_dict['standard_error'],
            min_max_score=result_dict['min_max_score'],
            estimator=self.get_saved_model(self.get_history_len()),
            iteration_number=result_dict['number_iterations']
        )

    def __get_dictionary_from_json(self, index):
        """
        Retorna um dicionário a partir do JSON do histórico

        :param index: Índice da lista de histórico que deseja recuperar
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"O arquivo {self.params_file_name}.json não foi encontrado no diretório {self.output_directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        if index < -1 or index >= len(data):
            raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        return result_dict

    def __save_model(self, estimator):
        """
        Função para salvar o modelo treinado e utilizá-lo para prever com outros dados.

        :param estimator: Estimador que deseja salvar
        """

        history_len = self.__get_history_len()
        output_path = os.path.join(self.output_directory, f"model_{history_len}.pkl")

        with open(output_path, 'wb') as file:
            pickle.dump(estimator, file)

    def get_saved_model(self, version: int):
        """
        Recupera o modelo que foi salvo de acordo com a versão

        :param version: Versão do modelo, concatenada no nome do arquivo, que deseja recuperar.
        """

        output_path = os.path.join(self.output_directory, f"model_{version}.pkl")

        with open(output_path, 'rb') as f:
            return pickle.load(f)

    def __get_history_len(self) -> int:
        """
        Retorna o tamanho da lista do histórico
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return 0

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data)
