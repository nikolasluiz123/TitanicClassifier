import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_processing import get_train_data
from hiper_params_search.grid_searcher import ClassifierGridHipperParamsSearcher
from manager.history_manager import BasicValidationHistoryManager
from manager.process_manager import ProcessManager
from model_validator.basic_validator import BasicValidator

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

y = df_train['sobreviveu']

search_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [2, 4, 6, 8],
    'max_features': [None, 'sqrt', 'log2'],
}

params_searcher = ClassifierGridHipperParamsSearcher(
    data_x=x,
    data_y=y,
    params=search_params,
    estimator=RandomForestClassifier()
)

validator = BasicValidator(data_x=x, data_y=y)
history_manager = BasicValidationHistoryManager(output_directory='history',
                                                models_directory='models_basic_validation_grid_search',
                                                params_file_name='tested_params_basic_validation_grid_search')

process_manager = ProcessManager(
    seed=42,
    params_searcher=params_searcher,
    validator=validator,
    history_manager=history_manager,
    save_history=True,
    history_index=None
)

process_manager.process()
