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

Essa estratégia é que gostumo mais utilizar, ela tem o ponto positivo de não testar todas as combinações possíveis, por isso, você pode adicionar todos os parâmetros do modelo e apenas limitar quantas vezes vai fazer o fit e procurar o melhor modelo. O ponto negativo é que
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

Após a busca dos parâmetros podemos utilizar a validação cruzada contina no [CrossValidator](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/cross_validator.py#L10). Internamente essa implementação utiliza [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html), resumindo o processo feito pela função, nós não separamos os dados em treino e teste, pegamos todos os dados e estabelecemos um número (5 por exemplo) e serão montados N grupos de 5 elementos, os quais chamamos de Folds. A partir do momento que foram estabilecidos os Folds um grupo será utilizado para treino, enquanto outro grupo será utilizado para teste, até que todas as combinações de grupos seja realizada e um conjunto de resultados (chamado de scores) seja retornado pela função.

Como nós temos N resultados podemos calcular mais métricas e armazarnar em [ClassifierCrossValidationResult](https://github.com/nikolasluiz123/TitanicClassifier/blob/master/model_validator/result.py#L15).

### Histórico


### Pipeline de Execução




