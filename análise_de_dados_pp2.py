# -*- coding: utf-8 -*-
"""Análise de Dados - PP2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1q6jsnyKCyLT3ubyHC1gzFKwB8SshZUUC

#**Aplicações de Tecnologias Emergentes**#

#**Parte 1**#
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

link = 'https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/d4332a3056f44e1a1dec9600a31f21c8_boston.csv'
boston = pd.read_csv(link)

boston

data = pd.DataFrame(data=boston)

data.head()

data.describe()

! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

from pandas_profiling import ProfileReport
profile = ProfileReport(data, title="Relatório - Pandas Profiling", html={'style':{'full_width':True}})

profile

profile.to_file(output_file="Relatório 01")

"""#**Parte 2**#"""

data.isnull().sum()

data.describe()

numeric_data = data.select_dtypes(include='number')  # Select only numeric columns
correlations = numeric_data.corr()  # Compute correlation matrix

plt.figure(figsize=(16,6))
sns.heatmap(data = correlations, annot=True)

import plotly.express as px
fig = px.scatter(data, x=data.RM, y= data.MEDV)
fig.show()

data.RM.describe()

import plotly.figure_factory as ff
labels = ["Distribuição da variável RM (Número de Quartos)"]
fig = ff.create_distplot([data.RM], labels, bin_size=-2)
fig.show()

import plotly.express as px
fig = px.box(data, y="RM")
fig.update_layout(width=800, height=800)
fig.show()

data.MEDV.describe()
import plotly.figure_factory as ff
labels = ["Distribuição da variável MEDV (preço médio do imóvel)"]
fig = ff.create_distplot([data.MEDV], labels, bin_size=-2)
fig.show()

from scipy import stats  # Corrected import
# Compute skewness of the MEDV column
skewness = stats.skew(data['MEDV'])
print(f"Skewness of MEDV: {skewness}")

# Plot histogram using Plotly
import plotly.express as px
fig = px.histogram(data, x="MEDV", nbins=50, opacity=0.50)
fig.show()

import plotly.express as px
fig = px.box(data, y="MEDV")
fig.update_layout(width=800, height=800)
fig.show()

data[['RM', 'MEDV', 'PTRATIO']].describe()

data[['RM', 'MEDV', 'PTRATIO']].nlargest(20, "MEDV")
top20 = data.nlargest(20, "MEDV").index
top20
data.drop(top20, inplace=True)

top16 = data[['RM', 'MEDV', 'PTRATIO']].nlargest(16, "MEDV").index
data.drop(top16, inplace=True)
import plotly.figure_factory as ff
labels = ['Distribuição da Variável MEdv (número de quartos)']
fig = ff.create_distplot([data.MEDV], labels, bin_size=-2)
fig.show()

fig = px.histogram(data, x="MEDV", nbins=50, opacity=0.50)
fig.show()

stats.skew(data.MEDV)

data.RM = data.RM.astype(int)

data.RM.describe()

categorias = []

for idx, valor in data['RM'].items():
    if valor <= 4:
        categorias.append("Pequeno")
    elif valor < 7:
        categorias.append("Médio")
    else:
        categorias.append("Grande")

print(categorias)

data['categorias'] = categorias
data.categorias.value_counts()
medias_categorias = data.groupby(by='categorias')['MEDV'].mean()
medias_categorias

dic_baseline = {'Grande':medias_categorias[0], 'Médio':medias_categorias[1], 'Pequeno':medias_categorias[2]}
dic_baseline

def retorna_baseline(num_quartos):
  if num_quartos <= 4:
    return dic_baseline.get('Pequeno')
  elif num_quartos < 7:
    return dic_baseline.get('Médio')
  else:
    return dic_baseline.get('Grande')

print(retorna_baseline(4))

for i in data.RM.items():
    n_quartos = i[1]
    print(f'O número de quartos é {n_quartos}, com o valor médio de {retorna_baseline(n_quartos)}')

"""#**Parte 3**#"""

y = data['MEDV']
x = data.drop(['TOWN','TRACT','LAT','LON','RAD','TAX','MEDV','DIS','AGE','ZN','categorias'], axis=1)
x.head()

y.head()

from sklearn.model_selection import train_test_split
x_train, x_teste, y_train, y_teste = train_test_split(x, y, test_size=0.2, random_state=5)
print(f'X_Train: Numero de Linhas e Colunas: {x_train.shape}')
print(f'X_Teste: Numero de Linhas e Colunas: {x_teste.shape}')
print(f'Y_Train: Numero de Linhas e Colunas: {y_train.shape}')
print(f'Y_Teste: Numero de Linhas e Colunas: {y_teste.shape}')

x_teste.head()

predicoes = []

for i in x_teste.RM.items():
  n_quartos = i[1]
  predicoes.append(retorna_baseline(n_quartos))
predicoes[:10]

df_results = pd.DataFrame()
df_results['valor_real'] = y_teste.values
df_results['valor_predito_baseline'] = predicoes[:len(y_teste)]
df_results.head(10)

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_results.index,
                         y= df_results.valor_real,
                         mode='lines+markers',
                         name="Valor Real"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y= df_results.valor_predito_baseline,
                         mode='lines+markers',
                         name="Valor Prédio Baseline"
))
fig.show()

print(len(y_teste))        # Deve retornar 94
print(len(predicoes))      # Deve retornar 94, caso contrário, haverá um erro

from sklearn.metrics import mean_squared_error
from math import sqrt
predicoes = predicoes[:len(y_teste)]

rmse = (np.sqrt(mean_squared_error(y_teste, predicoes)))
print("Peformance do Modelo BaseLine: ")
print(f"RMSE é {rmse}")

"""#**Machine Learning**#"""

!pip install scikit-learn

from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()

lin_model.fit(x_train, y_train)

# Supondo que X_teste seja o seu conjunto de dados de teste
y_pred = lin_model.predict(x_teste)

rmse = (np.sqrt(mean_squared_error(y_teste, y_pred)))

print("Peformance do Modelo avaliado com os dados do teste:")
print(rmse)

df_results["valor_predito_reg_linear"] = lin_model.predict(x_teste)

df_results.head(20)

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_real,
                         mode='lines+markers',
                         name="Valor Real"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_baseline,
                         mode='lines+markers',
                         name="Valor Baseline"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_baseline,
                         mode='lines',
                         line=dict(color="#FEBFB3"),
                         name="Valor Prédio Regressão Linear"
))
fig.show()

# Árvore de Decisão

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_teste)
df_results['valor_predito_arvore'] = y_pred
df_results.head(10)

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_real,
                         mode='lines+markers',
                         name="Valor Real"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_baseline,
                         mode='lines+markers',
                         name="Valor Baseline"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_baseline,
                         mode='lines+markers',
                         name="Valor Prédito Regressão Linear"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_arvore,
                         mode='lines+markers',
                         name="Valor Prédito Árvore"
))
fig.show()

rmse = (np.sqrt(mean_squared_error(y_teste, y_pred)))

print("Peformance do Modelo avaliado com os dados do teste:")
print(rmse)

# Random Forest

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(x_train, y_train)

y_pred = rf_regressor.predict(x_teste)
df_results['valor_predito_random_forest'] = y_pred
df_results.head(10)

rmse = (np.sqrt(mean_squared_error(y_teste, y_pred)))
print("Peformance do Modelo avaliado com os dados do teste:")
print(rmse)

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_real,
                         mode='lines+markers',
                         name="Valor Real"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_baseline,
                         mode='lines+markers',
                         name="Valor Baseline"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_baseline,
                         mode='lines+markers',
                         name="Valor Prédito Regressão Linear"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_arvore,
                         mode='lines+markers',
                         name="Valor Prédito Árvore"
))

fig.add_trace(go.Scatter(x=df_results.index,
                         y=df_results.valor_predito_random_forest,
                         mode='lines+markers',
                         name="Valor Prédito Random Forest"
))
fig.show()