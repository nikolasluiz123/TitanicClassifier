import pandas as pd
from tabulate import tabulate

df_treino = pd.read_csv('train.csv')

print('Dados de Treino:')
print(tabulate(df_treino.head(), headers='keys', tablefmt='psql'))
print()

df_treino.columns = ['id_passageiro', 'sobreviveu', 'classe_social', 'nome', 'sexo', 'idade', 'qtd_irmaos_conjuges',
                     'qtd_pais_filhos', 'ticket', 'valor_ticket', 'cabine', 'porta_embarque']

# Id do Passageiro, Nome do passageiro e ticket não fazem diferença para classificar
# Valor do ticket não faz diferença pois já temos a classe social. Talvez seja interessante testar com essa variável.
# Cabine possui muitos valores nulos. Acho que essa variável possa fazer diferença, talvez vale testar com ela
df_treino.drop(columns=['id_passageiro', 'nome', 'ticket', 'valor_ticket', 'cabine'], inplace=True, axis=1)

print('Info:')
print(df_treino.info())
print()

print('Pessoas com idade null:')
print(tabulate(df_treino[df_treino['idade'].isnull()], headers='keys', tablefmt='psql'))
print('Quantidade de Pessoas com idade null: ', df_treino[df_treino['idade'].isnull()].shape[0])
print()

# Remoção dos registros com idade null, talvez vale testar com esses registros depois
df_treino.dropna(subset=['idade'], inplace=True)

print('Dados de Treino Tratados:')
print(tabulate(df_treino.head(), headers='keys', tablefmt='psql'))
print()