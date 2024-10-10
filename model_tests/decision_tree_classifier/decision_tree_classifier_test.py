import pandas as pd
from scipy.stats import uniform, randint
from sklearn.model_selection import KFold

from data_processing import get_train_data
from hiper_params_search.random_search import DecisionTreeRandomSearchClassifierSearch

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

y = df_train['sobreviveu']

search_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': randint(1, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
    'max_features': [None, 'sqrt', 'log2'],
}

hiper_parameter_searcher = DecisionTreeRandomSearchClassifierSearch(data_x=x,
                                                                    data_y=y,
                                                                    params=search_params,
                                                                    cv=KFold(10),
                                                                    n_jobs=-1,
                                                                    history_dir='tested_params',
                                                                    history_file='tested_params_file',
                                                                    use_history=False)

randomized_search_cv = hiper_parameter_searcher.search_hipper_parameters(number_iterations=5000)

cross_val_score_result = hiper_parameter_searcher.calculate_cross_val_score(searcher=randomized_search_cv)
cross_val_score_result.show_cross_val_metrics()

hiper_parameter_searcher.show_processing_time()