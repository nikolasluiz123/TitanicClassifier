## Estrutura do Projeto

O projeto está estruturado para trabalhar com os dados do Titanic e nele é possível implementar qualquer classificador da biblioteca Scikit-Learn, também seria possível implementar Regressores Lineares com a mesma estrutura, mas não é o foco do projeto.

### Exploração e Processamento dos Dados

Para a fase inicial de exploração e processamento dos dados temos o diretório **data**. A exploração dos dados está localizada em [data_exploration](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/data/data_exploration.py) e o processamento para obtenção 
dos objetos DataFrame está localizada em [data_processing](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/data/data_processing.py).

### Seleção das Melhores Features

No projeto temos duas implementações para realizar a busca das features baseadas em [SelectKBest](https://scikit-learn.org/dev/modules/generated/sklearn.feature_selection.SelectKBest.html) e
[RFECV](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_selection.RFECV.html). 

A implementação SelectKBest foi implementada na classe [SelectKBestFeatureSearcher](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/regression_vars_search/k_best_feature_searcher.py#L8)
e RFECV foi implementada na classe [RecursiveFeatureSearcher](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/regression_vars_search/recursive_feature_searcher.py#L8).

SelectKBestFeatureSearcher utiliza uma estratégia com estatística e pergunta quantas features você deseja, essa implementação é mais rápida.
RecursiveFeatureSearcher utiliza uma estratégia de busca por exaustão e pergunta quantas features no mínimo você deseja, não é possível controlar quantas features exisitirão exatamente e é um processo mais lento.

### Busca de Hiper Parâmetros

A busca de parâmetros para os modelos é algo muito importante para obtermos bons resultados, o projeto mantem as implementações referentes a isso no diretório **hiper_params_search**. A biblioteca fornece várias formas
de busca dos melhores parâmetros do estimador, nesse projeto foi optado por utilizar a implementação [RandomizedSearchCV](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
implementada na classe [RandomHipperParamsSearcher](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/hiper_params_search/random_searcher.py#L5)

#### Busca Aleatória

Abaixo temos um exemplo de declaração dos parâmetros que deseja buscar valores:

```
search_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': randint(1, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
    'max_features': [None, 'sqrt', 'log2'],
}
```
Acima são declarados parâmetros para o [DecisionTreeClassifier](https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html), repare que alguns valores são listas finitas e alguns 
são chamadas de funções que retornam números inteiros ou decimais dentro de um intervalo delimitado. Veja que o parâmetro ``max_depth`` está utilizando um ``randint(1, 10)``, então em cada chamada dentro da implementação [RandomSearchCV](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
será atribuído para esse parâmetro do modelo um número inteiro aleatório entre 1 e 10.

Além dessa peculiaridade de podermos utilizar funções que retornam valores em um range nós também limitamos quantas vezes a implementação de busca vai até esse grid buscar valores e fazer o fit, normalmente chamamos isso de **número de iterações**.

Essa estratégia é que gosto mais utilizar, ela tem o ponto positivo de não testar todas as combinações possíveis, por isso, você pode adicionar todos os parâmetros do modelo e apenas limitar quantas vezes vai fazer o fit e procurar o melhor modelo. O ponto negativo é que
você não testará todas as combinações possíveis e talvez você não consiga encontrar o melhor modelo real, além de que, ao utilizar as funções que retornam números aleatórios (o que não é algo obrigatório), você pode acabar tendo resultados levemente diferentes entre as execuções.

### Validação do Melhor Modelo

No projeto foi optado por realizar a validação cruzada utilizando [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
dentro da implementação [ScikitLearnBaseValidator](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/validator.py#L15). Foi optado por esse tipo de validação
por conta da sua robustez, os dados são separados em grupos (folds) e a quantidade de dados dentro de cada fold é definida por nós.

Tendo sido realizada essa divisão dos folds ocorrerão diversos fits, a cada iteração um grupo é selecionado para teste e
o restante para treino, o processo acaba quando todos os folds já tiverem sido utilizados para teste uma vez. Como estamos falando 
de um processo de várias fits e predicts teremos uma lista que chamamos de scores, com eles podemos calcular mais métricas 
e armazarnar em [ScikitLearnCrossValidationResult](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/result.py#L18).

Foi implementada uma validação específica para classificação contendo duas coisas bem relevantes que o scikit-learn possui,
são elas [confusion_matrix](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.confusion_matrix.html) e
[classification_report](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html). Essas métricas são todas
calculadas e exibidas dentro da implementação [ClassifierFinalValidator](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/validator.py#L43).

### Histórico

O projeto possui uma implementação de histórico chamada [CrossValidationHistoryManager](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/history_manager.py#L169)
e o intuito dessa implementação é salvar os resultados das execuções e o modelo em si, para que possa ser reproduzir o mesmo resultado final
sem executar novamente.

### Gerenciador de Processos e Pipelines

Agora entramos na parte mais alto nível da implementação, basicamente essa implementação utiliza internamente tudo que
já foi explicado acima.

A classe [ScikitLearnMultiProcessManager](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/multi_process_manager.py#L281) é quem executa os
pipelines definidos, sendo que é possível definir um ou mais.

A implementação de pipeline existente no projeto é [ScikitLearnPipeline](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/multi_process_manager_pipelines.py#L48), nela é possível definir algumas coisas:

1. Definir qual estimador deseja testar.
2. Definir quais os parâmetros serão testados.
3. Implementação de busca das melhores features.
4. Implementação de busca de parâmetros.
5. Implementação da validação.
6. Implementação de gerenciamento de histórico.

Tendo definido a lista de pipelines ou ao menos um isso pode ser passado a implementação do ProcessManager juntamente
com outras definições que são globais e não são realizadas por pipeline. 

Ao fim de toda a execução o ProcessManager retornará uma lista contendo os melhores estimadores, os resultados dessa
lista vão depender de quais modelos você definiu nos pipelines. Além dessa lista o manager também vai salvar separadamente
o melhor dos melhores.

### Conclusão

Para que você possa ter a visão do processo como um todo lhe convido a utilizar o [Google Colab](https://colab.research.google.com/drive/1o64ErdHz1N5m_p55xPemPYLKC2aHhM9a?usp=sharing) do projeto