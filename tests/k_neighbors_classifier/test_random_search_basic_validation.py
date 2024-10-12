import pandas as pd
from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier

from data_processing import get_train_data
from hiper_params_search.random_searcher import ClassifierRandomHipperParamsSearcher
from manager.history_manager import BasicValidationHistoryManager
from manager.process_manager import ProcessManager
from model_validator.basic_validator import BasicValidator

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

y = df_train['sobreviveu']

search_params = {
    'n_neighbors': randint(1, 100),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}

params_searcher = ClassifierRandomHipperParamsSearcher(
    data_x=x,
    data_y=y,
    params=search_params,
    estimator=KNeighborsClassifier()
)

validator = BasicValidator(data_x=x, data_y=y)
history_manager = BasicValidationHistoryManager(output_directory='history',
                                                models_directory='models_basic_validation_random_search',
                                                params_file_name='tested_params_basic_validation_random_search')

process_manager = ProcessManager(
    seed=42,
    params_searcher=params_searcher,
    validator=validator,
    history_manager=history_manager,
    save_history=True,
    history_index=None
)

process_manager.process(number_interations=1000)
