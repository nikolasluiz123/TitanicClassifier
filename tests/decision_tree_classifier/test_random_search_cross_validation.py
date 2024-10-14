import pandas as pd
from scipy.stats import uniform, randint
from sklearn.tree import DecisionTreeClassifier

from data.data_processing import get_train_data
from hiper_params_search.random_searcher import ClassifierRandomHipperParamsSearcher
from hiper_params_search.result_analyzer import SearchResultAnalyzer
from manager.history_manager import CrossValidationHistoryManager
from manager.process_manager import ProcessManager
from model_validator.cross_validator import CrossValidator

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

y = df_train['sobreviveu']

search_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': randint(1, 100),
    'min_samples_split': randint(2, 80),
    'min_samples_leaf': randint(1, 30),
}

params_searcher = ClassifierRandomHipperParamsSearcher(
    data_x=x,
    data_y=y,
    params=search_params,
    estimator=DecisionTreeClassifier()
)

validator = CrossValidator(data_x=x, data_y=y)
history_manager = CrossValidationHistoryManager(output_directory='history',
                                                models_directory='models_cross_validation_random_search',
                                                params_file_name='tested_params_cross_validation_random_search')

analyzer = SearchResultAnalyzer(
    output_dir=r'history\scatter_plots\cross_validation_random_search',
    result_count=100,
    save=True
)

process_manager = ProcessManager(
    seed=42,
    params_searcher=params_searcher,
    validator=validator,
    history_manager=history_manager,
    result_analyzer=analyzer,
    save_history=True,
    history_index=None,
)

process_manager.process(number_interations=5000)
