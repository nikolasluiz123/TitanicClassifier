## Estrutura do Projeto

O projeto está estruturado para trabalhar com os dados do Titanic e nele é possível implementar qualquer classificador da biblioteca Scikit-Learn, também seria possível implementar Regressores Lineares com a mesma estrutura, mas não é o foco do projeto.

### Exploração e Processamento dos Dados

Para a fase inicial de exploração e processamento dos dados temos o diretório **data**. A exploração dos dados está localizada em [data_exploration](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/data/data_exploration.py) e o processamento para obtenção 
dos objetos DataFrame está localizada em [data_processing](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/data/data_processing.py).

### Busca de Hiper Parâmetros

A busca de parâmetros para os modelos é algo muito importante para obtermos bons resultados, o projeto mantem as implementações referentes a isso no diretório **hiper_params_search**. Como a biblioteca Scikit-Learn fornece duas formas de busca de parâmetros foi implementada
uma classe genérica para representar um 'Buscador de Parâmetros', a qual foi chamada de [ClassifierHipperParamsSearcher](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/hiper_params_search/searcher.py#L6), dessa forma podemos implementar tanto a Busca Aleatória
quanto a Busca em Grid.

#### Busca em Grid

Na busca utilizando a implementação [GridSearchCV](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html) você estabelece quais parâmetros você avaliará, assim como quais serão seus valores, tudo isso em um dicionário, por exemplo:

```
search_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6, 10],
    'max_features': [None, 'sqrt', 'log2', 4, 8, 16, 32],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4, 8, 16]
}
```
Acima são declarados alguns parâmetros do modelo [DecisionTreeClassifier](https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html), veja que para cada parâmetro é estabelecida uma lista finita de valores, sejam numéricos ou textos. Deve-se
tomar cuidado com a quantidade de parâmetros e valores que serão testados pois será feito o cálculo de combinações possíveis entre todos os valores, por exemplo: ``2 x 2 x 4 x 7 x 3 x 5 = 1680 combinações possíveis`` sendo esse o número de fits (treinamentos) que precisarão
ser realizados para que possa ser encontrado o melhor modelo.

Essa forma de realizar a busca tem como ponto positivo o fato de que todas as possibilidades serão testadas e o melhor modelo encontrado será de fato o melhor. O ponto negativo é que dependendo da quantidade de valores e parâmetros, além da complexidade dos dados de treino pode
fazer com que o tempo de processamento seja muito alto, por conta disso, quando utilizar essa estratégia, lembre-se de escolher bem quais valores testar.

No projeto temos [ClassifierGridHipperParamsSearcher](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/hiper_params_search/grid_searcher.py#L8) que utiliza ``GridSearchCV`` para buscar os parâmetros.

#### Busca Aleatória

Na busca aleatória o processo funciona um pouco diferente, vamos analisar a definição dos parâmetros:

```
search_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': randint(1, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
    'max_features': [None, 'sqrt', 'log2'],
}
```
Acima são declarados parâmetros para o mesmo modelo citado no tópico acima, repare que alguns valores ainda são listas finitas e alguns são chamadas de funções que retornam números inteiros ou decimais dentro de um intervalo delimitado. Essa é a diferença na parte da
declaração dos parâmetros que devem ser analisados, por exemplo, o parâmetro ``max_depth`` está utilizando um ``randint(1, 10)``, então em cada chamada dentro da implementação [RandomSearchCV](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
será atribuído para esse parâmetro do modelo um número inteiro aleatório entre 1 e 10.

Além dessa peculiaridade de podermos utilizar funções que retornam valores em um range nós também limitamos quantas vezes a implementação de busca vai até esse grid buscar valores e fazer o fit, normalmente chamamos isso de **número de iterações**.

Essa estratégia é que gosto mais utilizar, ela tem o ponto positivo de não testar todas as combinações possíveis, por isso, você pode adicionar todos os parâmetros do modelo e apenas limitar quantas vezes vai fazer o fit e procurar o melhor modelo. O ponto negativo é que
você não testará todas as combinações possíveis e talvez você não consiga encontrar o melhor modelo real.

No projeto temos [ClassifierRandomHipperParamsSearcher](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/hiper_params_search/random_searcher.py#L8) que utiliza ``RandomSearchCV`` para buscar os parâmetros.

### Validação do Melhor Modelo

Após a busca do melhor modelo utilizando uma das duas implementações citadas acima, precisamos validar a eficácia e ter certeza de que realmente ele é bom o suficiente para o que desejamos. No projeto temos o diretório **model_validator** que contem implementações
referentes a isso, especificamente falando de validadores temos [BaseValidator](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/validator.py#L8) que deve ser implementado por qualquer validador que desejarmos.

Da mesma forma que temos um **Validator** também temos um resultado dessa validação, contendo métricas matemáticas para indicar o quão bem o modelo foi treinado e lidará com novos dados. No projeto temos a implementação [ClassifierValidationResult](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/result.py#L4) que é implementada por todos os resultados.

#### Validação Simples

A validação simples inicia já na separação dos dados, quando desejar realizar esse tipo de validação a função [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) do Scikit-Learn pode ser utiliza, dessa
forma obteremos dados de treino e teste. Feito a separação dos dados, basta realizar um fit, seguido de um predict e calcular as métricas desejadas.

A implementação [BasicValidator](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/basic_validator.py#L10) possui apenas duas métricas: **acurácia** e **matriz de confusão** as quais são armazenadas no resultado [ClassifierBasicValidationResult](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/result.py#L62) juntamente do melhor modelo que foi encontrado na busca e validado.

Essa validação é uma boa forma de começar pois é simples de codificar e seu processamento costuma ser bem rápido, mas, o ponto negativo é que, como você realiza apenas uma divisão nos dados, treina e depois testa, você não consegue saber com tanta exatidão como o seu
modelo vai lidar com dados diferentes.

#### Validação Cruzada

Após a busca dos parâmetros podemos utilizar a validação cruzada contida no [CrossValidator](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/cross_validator.py#L10). Internamente essa implementação utiliza [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html), resumindo o processo feito pela função, nós não separamos os dados em treino e teste, pegamos todos os dados e estabelecemos um número (5 por exemplo) e serão montados N grupos de 5 elementos, os quais chamamos de Folds.

Tendo sido realizada essa divisão dos Folds ocorrerão diversos fits, a cada iteração um grupo é selecionado para teste e o restante para treino, o processo acaba quando todos os Folds já tiverem sido utilizados para teste uma vez. Como estamos falando de um processo de várias fits e predicts teremos uma lista que chamamos de scores, com eles podemos calcular mais métricas e armazarnar em [ClassifierCrossValidationResult](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/result.py#L15).

### Histórico

No projeto há uma implementação específica para lidar com o histórico das execuções e uma possível reutilização dos mesmos parâmetros que já foram processados em algum momento.

Quando realizamos a busca dos parâmetros salvamos os valores do melhor modelo, assim como as métricas obtidas na validação dele, tudo isso vai para uma lista em um arquivo JSON que pode ser recuperado através do índice. Vamos ver um exemplo:

Abaixo temos o JSON com apenas uma execução realizada da busca de parâmetros em conjunto com a validação básica
```
[
    {
        "score": 0.7486033519553073,
        "estimator_params": {
            "ccp_alpha": 0.0,
            "class_weight": null,
            "criterion": "entropy",
            "max_depth": 8,
            "max_features": "log2",
            "max_leaf_nodes": null,
            "min_impurity_decrease": 0.0,
            "min_samples_leaf": 6,
            "min_samples_split": 8,
            "min_weight_fraction_leaf": 0.17216952883254816,
            "monotonic_cst": null,
            "random_state": null,
            "splitter": "random"
        },
        "confusion_matrix": [
            [
                81,
                22
            ],
            [
                23,
                53
            ]
        ],
        "search_time": "00:00:03",
        "validation_time": "00:00:01"
    }
]
```

Veja que ``score``, ``estimator_params`` e ``confusion_matrix`` são referentes ao resultado dos processamentos de busca de parâmetro e da validação realizada, no caso, a básica. Os últimos campos servem para deixar registrado quanto tempo foi gasto para obter esse resultado.

Agora vejamos um JSON que representa a busca de parâmetros juntamente com a validação cruzada:

```
[
    {
        "mean": 0.7395055648576776,
        "standard_deviation": 0.05819282504690473,
        "median": 0.7482517482517482,
        "variance": 0.003386404886939662,
        "standard_error": 0.02602462252152627,
        "min_max_score": [
            0.6293706293706294,
            0.7972027972027972
        ],
        "estimator_params": {
            "ccp_alpha": 0.0,
            "class_weight": null,
            "criterion": "gini",
            "max_depth": 4,
            "max_features": null,
            "max_leaf_nodes": null,
            "min_impurity_decrease": 0.0,
            "min_samples_leaf": 16,
            "min_samples_split": 8,
            "min_weight_fraction_leaf": 0.13648244121947617,
            "monotonic_cst": null,
            "random_state": null,
            "splitter": "random"
        },
        "search_time": "00:00:03",
        "validation_time": "00:00:13"
    }
]
```

Basicamente segue o mesmo padrão, iniciamos com as métricas obtidas da validação cruzada, depois os parâmetros que geraram o modelo que trouxe esses resultados e, por fim, o tempo gasto para chegar nisso. Além de manipular esse histórico em arquivos JSON a implementação também é capaz de salvar o melhor modelo em um arquivo de extensão pkl que pode ser reutilizado.

A implementação base é [HistoryManager](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/history_manager.py#L10), sendo que as únicas funcionalidades que precisam de fato ser implementadas pelas classes filhas são as que lidam com os dados que deseja salvar, pois como vimos acima, as métricas variam e precisamos de classes específicas para transformar o dicionário python em JSON e transformar o JSON no objeto de resultado de validação específico.

Como possuímos a validação básica e a cruzada foram implementadas as classes [BasicValidationHistoryManager](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/history_manager.py#L190) e [CrossValidationHistoryManager](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/history_manager.py#L150)

### Pipeline de Execução

Agora que foram abordadas as implementações das operações específicas necessárias para a obtenção de um modelo de machine learning, será detalhada a classe que é responsável por centralizar todas essas implementações. No diretório **manager** há a classe [ProcessManager](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/process_manager.py#L9) a qual solicita
as 3 implementações já detalhadas: [ClassifierHipperParamsSearcher](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/hiper_params_search/searcher.py#L6), [BaseValidator](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/validator.py#L8) e [HistoryManager](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/history_manager.py#L10) e internamente na função principal dele, chamada [process](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/manager/process_manager.py#L30) são realizadas as chamadas dos processos que seguem uma ordem lógica definida.

Primeiro ocorre a busca do melhor modelo, em seguida ele é validado e, por fim, ele pode ser salvo no histórico se for desejado. Além disso, os processos de busca e validação do modelo utilizam bastante o esquema randômico e essa implementação já define uma seed do próprio numpy, isso faz com que não seja preciso passar para cada classe do Scikit-Learn um random_state.

## Testes Realizados

O objetivo principal era classificarmos se uma pessoa sobreviveria no Titanic baseado nos dados utilizando o algorítmo [KNeighborsClassifier](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), porem, achei que seria muito mais interessante se fossem realizados testes utilizando outros dois algorítmos que utilizam árvores, são eles [DecisionTreeClassifier](https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html) e [RandomForestClassifier](https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Além de diversificar os algorítmos utilizados também achei interessante que fosse utilizada a busca de parâmetros, tanto com a implementação de Grid quanto a Random, dessa maneira, foi possível otimizar consideravelmente os resultados dos modelos. Por fim, referente a validação, foi realizada uma validação básica, a qual foi solicitada, assim como também foi realizada uma validação mais robusta, para que possamos perceber as diferenças entre elas.

Dentro do diretório **tests** há o diretório **history**, o qual contem tanto os diretórios com os modelos, quanto os arquivos JSON com as listas de parâmetros utilizados e suas métricas.

## Conclusão

Após analisar os resultados dos algorítmos testados e das estratégias de busca de parâmetros e validação, acredito que, independete do algorítmo, as melhores combinações sempre serão utilizando [RandomizedSearchCV](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) em conjunto com a função [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) quando utilizada a biblioteca Scikit-Learn. A busca aleatória poupa muito tempo de processamento e entrega modelos muito bons, a validação cruzada, por sua vez, possibilita calcularmos muito mais métricas e termos muito mais certeza de como o modelo vai funcionar.

Agora referente aos três algorítmos testados, vamos analisar apenas os três arquivos JSONs: [Testes de DecisionTreeClassifier](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/tests/decision_tree_classifier/history/tested_params_cross_validation_random_search.json), [Testes de KNeighborsClassifier](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/tests/k_neighbors_classifier/history/tested_params_cross_validation_random_search.json) e [Testes de RandomForestClassifier](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/tests/random_forest_classifier/history/tested_params_cross_validation_random_search.json), todos eles com busca aleatória de parâmetros e validação cruzada.

As métricas matemáticas que particularmente acho mais relevantes são:

**Média** para ter uma noção mais básica de como foi a acurácia.

**Desvio Padrão** para entender se os parâmetros escolhidos para a busca precisam ser ajustados. Se esse número tiver muito alto as tentativas resultaram em valores distantes da média.

**Mínimo e Máximo** para traduzir um pouco o desvio padrão, desse jeito se o mínimo estiver muito baixo, foram testadas combinações de parâmetros ruins e talvez isso possa ser otimizado.

| Modelo                 | Média | Desvio Padrão | Acurácia Mínima | Acurácia Máxima | Tempo da Busca de Params | Tempo da Validação |
|------------------------|-------|---------------|-----------------|-----------------|--------------------------|--------------------|
| DecisionTreeClassifier | 0.74  | 0.058         | 0.63            | 0.80            | 00:00:03                 | 00:00:13           |
| KNeighborsClassifier   | 0.79  | 0.041         | 0.73            | 0.85            | 00:00:04                 | 00:00:27           |
| RandomForestClassifier | 0.78  | 0.038         | 0.72            | 0.84            | 00:00:24                 | 00:03:34           |

Avaliando a média, podemos notar que **RandomForestClassifier** e **KNeighborsClassifier** foram muito parecidos com os parâmetros testados.
O desvio padrão desses dois também são os menores e se parecem bastante.
Por fim, **KNeighborsClassifier** possui os valores mínimo e máximo mais altos, mas com pouca diferença.

Um critério que também pode ser levado em consideração para desempate seriam os tempos de execução, que também estão presentes no JSON. Se isso fosse considerado, **KNeighborsClassifier** poderia ser considerado como a melhor opção.

Caso você queira realizar seus testes utilizando o projeto como base, pode acessar o [arquivo compartilhado](https://colab.research.google.com/drive/1o64ErdHz1N5m_p55xPemPYLKC2aHhM9a?usp=sharing) do Google Colab que contem, de forma organizada, todo o código fonte do projeto.
