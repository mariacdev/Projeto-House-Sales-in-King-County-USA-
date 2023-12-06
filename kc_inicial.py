import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go
from matplotlib import gridspec

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
import pickle

data = pd.read_csv('kc_house_data.csv')
data.head()

#--------------------------------------------------------------------------------------------------------

def yesno(data):
   data = data.apply(lambda x: 'yes' if x> 0 else 'no')
   return(data)

def agrupamento(data, coluna):
  data = data[['price', coluna]].groupby(coluna).median().reset_index()
  return(data)

def correlacao(data, variavel):
   ziptest = data[data.zipcode == variavel]
   correlation = ziptest.corr()
   data = sn.heatmap(correlation, annot = True, fmt = ".1f", linewidths = .10)

def boxplot(coluna1):
   fig = px.box(data, x = coluna1, y = 'price')
   fig.show()

def histogram(data, x):
   fig = px.histograma(data, x=x, color="level_value", template = 'plotly_dark', darmode = 'group', color_discrete_sequence=['#AEFF02', '#031147'])
   fig.update_layout(
      paper_bgcolor = '#0D0C0C',
      plot_bgcolor = '0D0C0C',
      autosize = True)
   fig.show()

def layout(graph):
   graph.update_layout(
       paper_bgcolor = '#0D0C0C',
       plot_bgcolor = '#0D0C0C',
       autosize = True
 )
   graph.show()

#--------------------------------------------------------------------------------------------------------

data.dtypes
data.isnull().sum()
data.describe()
data['date'] = pd.to_datetime(data['date'])
data['zipcode'] = data['zipcode'].astype(str)
data = data.drop(['sqft_living15', 'sqft_lot15'], axis = 1 )

for i in range(len(data)):
   if data.loc[i,'bedrooms'] < 1 or data.loc[i,'bathrooms'] < 1:
     data = data.drop(i)
data = data.drop(15870).reset_index()
data = data.drop(['index'], axis = 1)

data['Basement?'] = yesno(data['sqft_basement'])
data['water_view'] = yesno(data['waterfront'])
data['Renoveted?'] = yesno(data['yr_renovated'])

data['Quartis_price'] = data['price'].apply(lambda x: 1 if x <= 323000 else
                                                  2 if (x > 450000) and (x <= 645000) else
                                                  3 if (x > 450000) and (x <= 645000) else
                                                  4 if (x>645000) and (x < 1127500) else 5)
data['sqft_level'] = data['sqft_living'].apply(lambda x: 1 if x <= 1430 else
                                                     2 if (x > 1430) and (x <= 1920) else
                                                     3 if (x > 1920) and (x <= 2550) else
                                                     4 if (x > 2550) and (x <= 4230) else 5)

data_zipcode = agrupamento(data, 'zipcode')

regiao = data_zipcode['zipcode'].unique()
for i in regiao:
   lista = data[data['zipcode']==i].index.tolist()
   num = data[data.zipcode == i].loc[:, 'price'].median()
   for m in lista:
      data.loc[m, 'level_value'] = 'Alto' if data[data.zipcode == i].loc[m, 'price'] > num else 'Baixo'

data_zipcode = data_zipcode.sort_values(by = ['price'])
data_zipcode['price'].mean()
data_zipcode1 = data_zipcode
for i in range(len(data_zipcode1)):
  if data_zipcode1.loc[i, 'price'] < 501607:
    data_zipcode1 = data_zipcode1.drop(i)
data_zipcode1 = data_zipcode1['zipcode'].values

data['zipcode_class'] = 'abaixo'
for m in data_zipcode1:
  for i in range (len(data)):
    if data.loc[i, 'zipcode'] == m:
      data.loc[i, 'zipcode_class'] = 'acima'

data_bath = agrupamento(data, 'bathrooms')
data_grade = agrupamento(data, 'grade')
data_view = agrupamento(data, 'view')
data_bedrooms = agrupamento(data, 'bedrooms')
data_condition = agrupamento(data, 'condition')
data_waterfront = agrupamento(data, 'water_view')

data_high_price = data[data.Quartis_price > 3]

data['mes_ano'] = data['date'].dt.strftime('%Y-%m')
by_date = data[['price', 'mes_ano']].groupby('mes_ano').median().reset_index()
by_date_Baixo = data[data.level_value == 'Baixo'][['price', 'mes_ano']].groupby('mes_ano').median().reset_index()
by_date_Alto = data[data.level_value == 'Alto'][['price', 'mes_ano']].groupby('mes_ano').median().reset_index()

by_yrbuilt_median = data[['price', 'yr_built']].groupby('yr_built').median().reset_index()
by_yrbuilt_Alto = data[data.level_value == 'Alto'][['price', 'yr_built' ]].groupby('yr_built').median().reset_index()
by_yrbuilt_Baixo = data[data.level_value == 'Baixo'][['price', 'yr_built' ]].groupby('yr_built').median().reset_index()

data['sazonal'] = data['mes_ano'].apply(lambda x: 'prima-verão' if (x < '2014-08') or (x> '2015-03') else
                                                   'Out-inver' )

sazonal = data[['price', 'sazonal']].groupby('sazonal').median().reset_index()

data.to_csv('kc_definitivo.csv', index=False)
#--------------------------------------------------------------------------------------------------------

figura = plt.figure(figsize=(20,20))
sn.heatmap(data.corr(), cmap="YlGnBu", annot=True)

boxplot('grade')
boxplot('condition')
boxplot('bathrooms')
boxplot('view')
boxplot('floors')
boxplot('bedrooms')

fig = px.box(data, x = 'price',
              template = 'plotly_dark',
              labels = {'price' : 'Preço'},
              color_discrete_sequence=['#AEFF02', '#031147'],
              )

layout(fig)

fig = px.box(data, x = 'sqft_living',
              template = 'plotly_dark',
              labels = {'sqft_living' : 'Área interna'},
              color_discrete_sequence=['#AEFF02', '#031147'],
              )
layout(fig)

fig = px.scatter(data, x="sqft_living", y='price', trendline = 'ols', trendline_color_override = '#EF1609', template = 'plotly_dark',labels={
    'price': 'Preço das casas',
    'sqft_living': 'Tamanho interno por pés²'},
    color_discrete_sequence=['#AEFF02', '#031147'],
    title ='Média dos precos das casas de acordo com a área interna'
 )
layout(fig)

fig = px.bar(data_bath, x="bathrooms",
             y='price',
             template = 'plotly_dark',
             color_discrete_sequence=['#AEFF02', '#031147'],
             labels={
                     'price': 'Preço das casas',
                     'bathrooms': 'Banheiros'},
             title ='Média dos preços das casas de acordo com o número de banheiros')
layout(fig)

fig = px.bar(sazonal, x="sazonal",
             y='price',
             template = 'plotly_dark',
             color_discrete_sequence=['#AEFF02', '#031147'],
             labels={
                     'price': 'Preço das casas',
                     'bathrooms': 'Banheiros'},
             title ='Média dos preços das casas de acordo com a sazonalidade', text = 'price')
layout(fig)

fig = px.histogram(data, x='grade', color="level_value",
                   template = 'plotly_dark',
                   barmode = 'group',
                   color_discrete_sequence=['#AEFF02', '#031147'],
                   labels = {'grade' : 'Avaliação', 'level_value': 'valor da casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o nivel da avaliação ')

fig = px.bar(data_view, x="view",
             y='price',
             template = 'plotly_dark',
             color_discrete_sequence=['#AEFF02', '#031147'],
             labels={
                     'price': 'Preço das casas',
                      'view': 'Vista da casa'},
             title ='Média dos preços das casas de acordo com o nivel da vista'
 )
layout(fig)

fig = px.bar(data_bedrooms, x="bedrooms",
             y='price',
             template = 'plotly_dark',
             color_discrete_sequence=['#AEFF02', '#031147'],
             labels={
                     'price': 'Preço das casas',
                      'bedrooms': 'Número de quartos da casa'},
             title ='Média dos preços das casas de acordo com o número de quartos'
 )
layout(fig)

fig = px.bar(data_condition, x="condition",
             y='price',
             template = 'plotly_dark',
             color_discrete_sequence=['#AEFF02', '#031147'],
             labels={
                     'price': 'Preço das casas',
                     'condition': 'Nível da condição'},
             title ='Média dos preços das casas de acordo com a condição'
 )
layout(fig)

fig = px.bar(data_waterfront, x="water_view",
             y='price',
             template = 'plotly_dark',
             color_discrete_sequence=['#AEFF02', '#031147'],
             labels={
                     'price': 'Preço das casas',
                     'water_view': 'Possui vista para o mar'},
             title ='Média dos preços considerando se a casa tem vista para o mar'
 )
layout(fig)

mapa_escala = px.scatter_mapbox(data, lat='lat',lon='long', hover_name = 'price',
                                 color = 'Quartis_price',
                                 labels = {'Quartis_price' : 'Níveis de preço'},
                                 title = 'Mapa com todas as casas, por cores que abrangem a escala de níveis de 1 a 5.',
                                 template = 'plotly_dark',
                                 color_continuous_scale=px.colors.sequential.gray_r,
                                 size_max=10,zoom=9)
mapa_escala.update_layout(mapbox_style = 'carto-darkmatter')
mapa_escala.update_layout(height = 700, width = 750, margin = {'r':0, 't':45, 'l':0, 'b':0})
mapa_escala.show()

histogram(data,'grade')
histogram(data,'bathrooms')
histogram(data,'bedrooms')
histogram(data,'condition')
histogram(data,'waterfront')
histogram(data,'view')
histogram(data,'Renoveted?')
histogram(data,'Basement?')
histogram(data,'sazonal')

fig = px.histogram(data, x='grade', color="level_value",
                   template = 'plotly_dark',
                   barmode = 'group',
                   color_discrete_sequence=['#AEFF02', '#031147'],
                   labels = {'grade' : 'Avaliação', 'level_value': 'valor da casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o nivel da avaliação ')

fig = px.histogram(data, x='bathrooms', color="level_value",
                   template = 'plotly_dark',
                   barmode = 'group',
                   width= 800,
                   color_discrete_sequence=['#AEFF02', '#031147'],
                   labels = {'bathrooms' : 'Número de Banheiros', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o número de banheiros')
fig.update_traces(textposition='inside',texttemplate='%{text:.2s}')
layout(fig)

fig = px.histogram(data, x='sqft_level', color="level_value",
                   template = 'plotly_dark',
                   barmode = 'group', width=760,
                   color_discrete_sequence=['#AEFF02', '#031147'],
                   labels = {'sqft_level' : 'Nível do tamanho interno', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o tamanho interno')
layout(fig)

fig = px.histogram(data, x='bedrooms', color="level_value",
                   template = 'plotly_dark',
                   barmode = 'group',
                   color_discrete_sequence=['#AEFF02', '#031147'],
                   labels = {'bedrooms' : 'Número de quartos', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o número de Quartos')
layout(fig)

fig = px.histogram(data, x='view', color="level_value",
                   template = 'plotly_dark',
                   barmode = 'group',
                   width=750,
                   color_discrete_sequence=['#AEFF02', '#031147'],
                   labels = {'view' : 'Nível da Vista', 'level_value': 'Valor da Casa'}, title = 'Quantidade de casas acima e abaixo da mediana de acordo com o nível da vista')
layout(fig)

fig = px.line(by_date_Baixo, x="mes_ano", y='price', title='Variação de preço das casas abaixo da média em suas regiões, durante o periodo de Maio de 2014 até maio de 2015')
fig.show()

fig = px.line(by_date_Alto, x="mes_ano", y='price', title='Variação de preço das casas acima da média em suas regiões, durante o periodo de Maio de 2014 até maio de 2015')
fig.show()

fig = px.line(by_yrbuilt_median, x="yr_built", y='price', title='Variação de preço de todas as casas, de acordo com seu ano de contrução')
fig.show()
fig = px.line(by_yrbuilt_Alto, x="yr_built", y='price', title='Variação de preço de todas as casas acima da mediana de suas regiões, de acordo com seu ano de contrução')
fig.show()
fig = px.line(by_yrbuilt_Baixo, x="yr_built", y='price', title='Variação de preço de todas as casas abaixo da mediana de suas regiões, de acordo com seu ano de contrução')
fig.show()

fig = px.line(by_date, x="mes_ano",
              y='price', width=850,
              color_discrete_sequence=['#AEFF02', '#031147'],
              template = 'plotly_dark',
              labels = {'price': 'Preço das Casas'},
              title='Variação de preço médio das casas, durante o periodo de Maio de 2014 até maio de 2015')
layout(fig)
