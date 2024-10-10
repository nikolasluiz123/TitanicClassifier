import json
import os

from sklearn.tree import DecisionTreeClassifier

from model_validation.classifier_validation import ClassifierCrossValScoreResult


class SearchParamsHistoryManager:

    def __init__(self, output_directory: str, file_name: str):
        self.output_directory = output_directory
        self.file_name = file_name

    def save_result(self, classifier_result: ClassifierCrossValScoreResult, search_time: str, validation_time: str):
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

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        output_path = os.path.join(self.output_directory, f"{self.file_name}.json")

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
        Verifica se o arquivo de histórico existe e contém pelo menos uma entrada.
        Retorna True se houver histórico, False caso contrário.
        """
        output_path = os.path.join(self.output_directory, f"{self.file_name}.json")

        if not os.path.exists(output_path):
            return False

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data) > 0

    def load_result_from_history(self, index: int = -1) -> ClassifierCrossValScoreResult:
        """
        Lê o histórico de um arquivo JSON e recria um objeto ClassifierCrossValScoreResult
        e um estimador DecisionTreeClassifier com os parâmetros do 'estimator_params'.
        """
        output_path = os.path.join(self.output_directory, f"{self.file_name}.json")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"O arquivo {self.file_name}.json não foi encontrado no diretório {self.output_directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        if index < -1 or index >= len(data):
            raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        estimator = DecisionTreeClassifier(**result_dict['estimator_params'])

        return ClassifierCrossValScoreResult(
            mean=result_dict['mean'],
            standard_deviation=result_dict['standard_deviation'],
            median=result_dict['median'],
            variance=result_dict['variance'],
            standard_error=result_dict['standard_error'],
            min_max_score=result_dict['min_max_score'],
            estimator=estimator,
            iteration_number=result_dict['number_iterations']
        )