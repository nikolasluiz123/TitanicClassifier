import warnings

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data.data_processing import get_train_data
from hiper_params_search.random_searcher import RandomHipperParamsSearcher
from manager.history_manager import CrossValidationHistoryManager
from manager.multi_process_manager import ScikitLearnPipeline, ScikitLearnMultiProcessManager
from model_validator.cross_validator import CrossValidatorScikitLearn
from model_validator.validator import ClassifierFinalValidator
from regression_vars_search.recursive_feature_searcher import RecursiveFeatureSearcher

warnings.filterwarnings("ignore", category=RuntimeWarning)

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns

x = pd.get_dummies(x, columns=obj_columns)
y = df_train['sobreviveu']

feature_searcher = RecursiveFeatureSearcher(log_level=1)
params_searcher = RandomHipperParamsSearcher(number_iterations=1000, log_level=1)
cross_validator = CrossValidatorScikitLearn(log_level=1)

pipelines = [
    ScikitLearnPipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2'],
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='decision_tree_classifier_models',
            params_file_name='decision_tree_classifier_best_params')
    ),
    ScikitLearnPipeline(
        estimator=RandomForestClassifier(),
        params={
            'n_estimators': randint(10, 50),
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='random_forest_classifier_models',
            params_file_name='random_forest_classifier_best_params')
    ),
    ScikitLearnPipeline(
        estimator=SVC(),
        params={
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': randint(2, 5),
            'gamma': ['scale', 'auto'],
            'shrinking': [True, False]
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='svc_models',
            params_file_name='svc_best_params')
    ),
    ScikitLearnPipeline(
        estimator=GaussianProcessClassifier(),
        params={
            'optimizer': ['fmin_l_bfgs_b', None],
            'n_restarts_optimizer': randint(0, 10),
            'multi_class': ['one_vs_rest', 'one_vs_one'],
            'max_iter_predict': [100, 300, 500, 700, 900]
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='gausian_process_classifier_models',
            params_file_name='gausian_process_classifier_best_params')
    ),
    ScikitLearnPipeline(
        estimator=KNeighborsClassifier(),
        params={
            'n_neighbors': randint(1, 10),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': randint(1, 100),
            'p': [1, 2],
            'metric': ['minkowski', 'euclidean', 'manhattan']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='k_neighbors_classifier_models',
            params_file_name='k_neighbors_classifier_best_params')
    ),
    ScikitLearnPipeline(
        estimator=MLPClassifier(),
        params={
            'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': uniform(loc=0.0001, scale=0.01),
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='mlp_classifier_models',
            params_file_name='mlp_classifier_best_params')
    ),
    ScikitLearnPipeline(
        estimator=AdaBoostClassifier(),
        params={
            'n_estimators': randint(10, 50),
            'learning_rate': uniform(loc=0.01, scale=0.5),
            'algorithm': ['SAMME', 'SAMME.R']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='ada_boost_classifier_models',
            params_file_name='ada_boost_classifier_best_params')
    )
]

best_params_history_manager = CrossValidationHistoryManager(output_directory='history_bests',
                                                            models_directory='best_models',
                                                            params_file_name='best_params')
manager = ScikitLearnMultiProcessManager(
    data_x=x,
    data_y=y,
    seed=42,
    fold_splits=10,
    pipelines=pipelines,
    history_manager=best_params_history_manager,
    save_history=True,
)

manager.process_pipelines()

best_estimator = best_params_history_manager.load_validation_result_from_history().estimator
final_validator = ClassifierFinalValidator(estimator=best_estimator, data_x=x, data_y=y)
final_validator.validate()
