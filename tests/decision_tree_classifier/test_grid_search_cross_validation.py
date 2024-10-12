import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from data.data_processing import get_train_data
from hiper_params_search.grid_searcher import ClassifierGridHipperParamsSearcher
from manager.history_manager import CrossValidationHistoryManager
from manager.process_manager import ProcessManager
from model_validator.cross_validator import CrossValidator

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

y = df_train['sobreviveu']

search_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6, 10],
    'max_features': [None, 'sqrt', 'log2', 4, 8, 16, 32],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4, 8, 16]
}

params_searcher = ClassifierGridHipperParamsSearcher(
    data_x=x,
    data_y=y,
    params=search_params,
    estimator=DecisionTreeClassifier()
)

validator = CrossValidator(data_x=x, data_y=y)
history_manager = CrossValidationHistoryManager(output_directory='history',
                                                models_directory='models_cross_validation_grid_search',
                                                params_file_name='tested_params_cross_validation_grid_search')

process_manager = ProcessManager(
    seed=42,
    params_searcher=params_searcher,
    validator=validator,
    history_manager=history_manager,
    save_history=True,
    history_index=None
)

process_manager.process()
