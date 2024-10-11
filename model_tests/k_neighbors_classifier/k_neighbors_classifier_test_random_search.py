import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate

from data_processing import get_train_data, get_test_data
from hiper_params_search.random_search import ClassifierRandomHipperParamsSearch

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

y = df_train['sobreviveu']

search_params = {
    'n_neighbors': randint(1, 10),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}

hiper_parameter_searcher = ClassifierRandomHipperParamsSearch(data_x=x,
                                                              data_y=y,
                                                              params=search_params,
                                                              cv=KFold(10),
                                                              n_jobs=-1,
                                                              history_dir='tested_params',
                                                              history_file='tested_params_file',
                                                              use_history=True,
                                                              estimator=KNeighborsClassifier())

randomized_search_cv = hiper_parameter_searcher.search_hipper_parameters(number_iterations=100)

cross_val_score_result = hiper_parameter_searcher.calculate_cross_val_score(searcher=randomized_search_cv)
cross_val_score_result.show_cross_val_metrics()

hiper_parameter_searcher.show_processing_time()

df_test = get_test_data()

obj_columns = df_test.select_dtypes(include='object').columns
x_test = pd.get_dummies(df_test, columns=obj_columns)

model = hiper_parameter_searcher.history_manager.get_saved_model(version=1)

y_pred = model.predict(x_test)

df_test['previsao'] = y_pred

print(tabulate(df_test, headers='keys', tablefmt='psql'))
