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
      plot_bgcolor = '#0D0C0C',
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

#--------------------------------------------------------------------------------------
df = pd.read_csv('kc_definitivo.csv') 

data = data.drop(['lat'], axis = 1)
data = data.drop(['long'], axis = 1)
data = data.drop(['sqft_above'], axis = 1)
data = data.drop(['sqft_basement'], axis = 1)
data = data.drop(['level_value'], axis = 1)
data = data.drop(['sqft_lot'], axis = 1)
data = data.drop(['waterfront'], axis = 1)
data = data.drop(['Quartis_price'], axis = 1)
data = data.drop(['sqft_level'], axis = 1)
data = data.drop(['zipcode_class'], axis = 1)

data['zipcode'] = data['zipcode'].astype(str)

label_encoder_zipcode = LabelEncoder()
label_encoder_Renovated = LabelEncoder()
label_encoder_Basement = LabelEncoder()
label_encoder_mes_ano = LabelEncoder()
label_encoder_sazonal = LabelEncoder()
label_encoder_waterfront = LabelEncoder()

x_kc = data.iloc[:, 3:18].values
y_kc = data.iloc[:, 2].values

x_kc[:,9] = label_encoder_zipcode.fit_transform(x_kc[:,9])
x_kc[:,10] = label_encoder_Basement.fit_transform(x_kc[:,10])
x_kc[:,11] = label_encoder_waterfront.fit_transform(x_kc[:,11])
x_kc[:,12] = label_encoder_Renovated.fit_transform(x_kc[:,12])
x_kc[:,13] = label_encoder_mes_ano.fit_transform(x_kc[:,13])
x_kc[:,14] = label_encoder_sazonal.fit_transform(x_kc[:,14])

OneHotEncoder_hr = ColumnTransformer(transformers=[('Onehot', OneHotEncoder(), [9,10,11,12,13,14])], remainder = 'passthrough')
x_kc = OneHotEncoder_hr.fit_transform(x_kc)
x_kc = x_kc.toarray()

x_kc_treinamento, x_kc_teste, y_kc_treinamento, y_kc_teste = train_test_split(x_kc, y_kc, test_size = 0.15, random_state = 0)

scaler_x_kc = StandardScaler()
x_kc_treinamento_scaled = scaler_x_kc.fit_transform(x_kc_treinamento)
scaler_y_kc = StandardScaler()
y_kc_treinamento_scaled = scaler_y_kc.fit_transform(y_kc_treinamento.reshape(-1,1))

x_kc_teste_scaled = scaler_x_kc.transform(x_kc_teste)
y_kc_teste_scaled = scaler_y_kc.transform(y_kc_teste.reshape(-1,1))

regressor_multiplo_kc = LinearRegression()
regressor_multiplo_kc.fit(x_kc_treinamento, y_kc_treinamento)
regressor_multiplo_kc.score(x_kc_treinamento, y_kc_treinamento)
regressor_multiplo_kc.score(x_kc_teste, y_kc_teste)
previsoes = regressor_multiplo_kc.predict(x_kc_teste)
mean_absolute_error(y_kc_teste, previsoes)

poly = PolynomialFeatures(degree = 2)
x_kc_treinamento_poly = poly.fit_transform(x_kc_treinamento)
x_kc_teste_poly = poly.transform(x_kc_teste)

regressor_kc_poly = LinearRegression()
regressor_kc_poly.fit(x_kc_treinamento_poly, y_kc_treinamento)                      
regressor_kc_poly.score(x_kc_treinamento_poly, y_kc_treinamento)
regressor_kc_poly.score(x_kc_teste_poly, y_kc_teste)                     
previsoes_poly = regressor_kc_poly.predict(x_kc_teste_poly)
previsoes_poly         
mean_absolute_error(y_kc_teste, previsoes_poly)    

regressor_arvore_kc = DecisionTreeRegressor()
regressor_arvore_kc.fit(x_kc_treinamento, y_kc_treinamento)
regressor_arvore_kc.score(x_kc_treinamento, y_kc_treinamento)
regressor_arvore_kc.score(x_kc_teste, y_kc_teste)
previsoes_arvore = regressor_arvore_kc.predict(x_kc_teste)
mean_absolute_error(y_kc_teste, previsoes_arvore)

regressor_random_kc = RandomForestRegressor(n_estimators = 100)
regressor_random_kc.fit(x_kc_treinamento, y_kc_treinamento)
regressor_random_kc.score(x_kc_treinamento, y_kc_treinamento)
regressor_random_kc.score(x_kc_teste, y_kc_teste)
previsoes_random = regressor_random_kc.predict(x_kc_teste)
mean_absolute_error(y_kc_teste, previsoes_random)

regressor_svr_kc = SVR(kernel='rbf')
regressor_svr_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_svr_kc.score(x_kc_treinamento_scaled, y_kc_treinamento_scaled)
regressor_svr_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_svr_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1, 1)
y_kc_teste_inverse = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse, previsoes_inverse)

regressor_rna_kc = MLPRegressor()
regressor_rna_kc.fit(x_kc_treinamento_scaled, y_kc_treinamento_scaled.ravel())
regressor_rna_kc.score(x_kc_treinamento_scaled, y_kc_treinamento_scaled)
regressor_rna_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_rna_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1,1)
y_kc_teste_inverse_rna = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse_rna = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse_rna, previsoes_inverse_rna)

primeiros_resultados = {'Modelos' : ['Regressão Linear', 'Regressão polinominal','Árvore de decisão', 'Random Forest Regressor', 'SVR', 'Rede neural'],
'score treinamento': [0.81223, 0.92051, 0.99997, 0.97751, 0.85804, 0.95287 ],
'score test': [0.80018, -196.89182, 0.62866, 0.79508, 0.79508, 0.85604],
'mean absolute error': [94535.60, 210963.39, 115285.45, 82768.32, 75399.06, 86902.68]}

primeiros_resultados = pd.DataFrame(primeiros_resultados)

#--------------------------------------------------------------------------------------

x_kc_house = np.concatenate((x_kc_treinamento, x_kc_teste), axis = 0)
y_kc_house = np.concatenate((y_kc_treinamento, y_kc_teste), axis = 0)
x_kc_scaled = np.concatenate((x_kc_treinamento_scaled, x_kc_teste_scaled), axis = 0)
y_kc_scaled = np.concatenate((y_kc_treinamento_scaled, y_kc_teste_scaled), axis = 0)

parametros_random = {'n_estimators': [10,50,100,150],
             'min_samples_split' : [2,5,10],
             'min_samples_leaf' : [1, 5, 10],
             'n_jobs' : [-1]}

grid_search = GridSearchCV(estimator = RandomForestRegressor(), param_grid = parametros_random)
grid_search.fit(x_kc_house, y_kc_house)
melhores_parametros = grid_search.best_params_
melhor_score = grid_search.best_score_
print(melhores_parametros)
melhor_score
parametros_svr = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}
y_kc_scaled = y_kc_scaled.ravel()

grid_search = GridSearchCV(estimator = SVR(), param_grid = parametros_svr)
grid_search.fit(x_kc_scaled, y_kc_scaled)

melhores_parametros_svr = grid_search.best_params_
melhor_score_svr = grid_search.best_score_
print(melhores_parametros_svr)
melhor_score_svr

parametros_mlp = {'activation' : ['logistic', 'tanh', 'relu'],
                 'solver' : ['adam', 'sgd'],
                 'batch_size' : [10, 56]}

grid_search = GridSearchCV(estimator = MLPRegressor(), param_grid = parametros_mlp)
grid_search.fit(x_kc_scaled, y_kc_scaled)

melhores_parametros_mlp = grid_search.best_params_
melhor_score_mlp = grid_search.best_score_
print(melhores_parametros_mlp)
melhor_score_mlp


resultados_random_forest = []
resultados_svm = []
resultados_rede_neural = []

for i in range(30):
    kfold = KFold(n_splits = 10, shuffle = True, random_state = i)

    random_forest = RandomForestRegressor(min_samples_leaf = 1,
                                          min_samples_split= 5,
                                          n_estimators= 100,
                                          n_jobs = -1)
    scores = cross_val_score(random_forest, x_kc_house, y_kc_house, cv = kfold)
    resultados_random_forest.append(scores.mean())
    
    svm = SVR(kernel = 'poly')   
    scores = cross_val_score(svm, x_kc_scaled, y_kc_scaled, cv = kfold)
    resultados_svm.append(scores.mean())
    
    MLP_Regressor = MLPRegressor(activation= 'relu',
                                 batch_size= 56,
                                 solver = 'sgd')   
    scores = cross_val_score(MLP_Regressor, x_kc_scaled, y_kc_scaled, cv = kfold)
    resultados_rede_neural.append(scores.mean()) 

resultados = pd.DataFrame()
resultados = pd.DataFrame({'Random Forest': resultados_random_forest,
                           'SVM': resultados_svm, 
                           'Rede neural': resultados_rede_neural})
resultados.describe()

shapiro(resultados_random_forest), shapiro(resultados_svm), shapiro(resultados_rede_neural)

_, p = f_oneway(resultados_random_forest, resultados_svm, resultados_rede_neural)

resultados_algoritmos = {'accuracy': np.concatenate([resultados_random_forest,resultados_svm, resultados_rede_neural]),
                         'algoritmo': [
                          'random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest',
                          'random_forest','random_forest','random_forest','random_forest',
                          'svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm',
                          'rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural']}

resultados_data = pd.DataFrame(resultados_algoritmos)
resultados_data

compara_algoritmos = MultiComparison(resultados_data['accuracy'], resultados_data['algoritmo'])
teste_estatistico = compara_algoritmos.tukeyhsd()
print(teste_estatistico)

regressor_rna_kc = MLPRegressor(activation= 'relu',batch_size= 56, solver = 'sgd',max_iter=1000, hidden_layer_sizes=(9,9))
regressor_rna_kc.fit(x_kc_scaled, y_kc_scaled.ravel())
regressor_rna_kc.score(x_kc_scaled, y_kc_scaled)
regressor_rna_kc.score(x_kc_teste_scaled, y_kc_teste_scaled)
previsoes = regressor_rna_kc.predict(x_kc_teste_scaled)
previsoes = previsoes.reshape(-1,1)
y_kc_teste_inverse_rna = scaler_y_kc.inverse_transform(y_kc_teste_scaled)
previsoes_inverse_rna = scaler_y_kc.inverse_transform(previsoes)
mean_absolute_error(y_kc_teste_inverse_rna, previsoes_inverse_rna)

pickle.dump(regressor_rna_kc, open ('rede_neural_finalizado.sav', 'wb'))

rede_neural = pickle.load(open('rede_neural_finalizado.sav','rb'))
# --------------------------------------------------------------------------------------

data.loc[1,'price']

scaler_x_kc = StandardScaler()
x_kc_scaled = scaler_x_kc.fit_transform(x_kc)
scaler_y_kc = StandardScaler()
y_kc_scaled = scaler_y_kc.fit_transform(y_kc.reshape(-1,1))

regressor_rna_kc = MLPRegressor(activation= 'relu', batch_size= 56, solver = 'sgd',max_iter=1000, hidden_layer_sizes=(9,9))
regressor_rna_kc.fit(x_kc_scaled, y_kc_scaled.ravel())
regressor_rna_kc.score(x_kc_scaled, y_kc_scaled)
y_kc_scaled = scaler_y_kc.inverse_transform(y_kc_scaled)

x_kc.shape
x_kc_scaled
data_definitivo= np.concatenate((x_kc_scaled, data.iloc[:,0:3]), axis = 1)

# data_definitivo = pd.DataFrame(data_definitivo)

testes = regressor_rna_kc.predict(x_kc_scaled)
testes = testes.reshape(1,-1)
testes = scaler_y_kc.inverse_transform(testes)
testes = testes.ravel()
testes = pd.DataFrame(testes)

data_definitivo[103]= testes

for i in range(len(data_definitivo)):
  if data_definitivo.loc[i,102] < data_definitivo.loc[i,103]:
     data_definitivo.loc[i,104] = 'Abaixo'
  else:
      data_definitivo.loc[i,104] = 'Acima'  

data.iloc[:,17:19]
data_definitivo= np.concatenate((data_definitivo, data.iloc[:,17:19]), axis = 1)

data_definitivo = pd.DataFrame(data_definitivo)

mapa_final = px.scatter_mapbox(data_definitivo, lat=105,lon=106,
                                 color = 'valor_casa', hover_name = 'price', 
                                 labels = {'104' : 'Níveis de preço'}, 
                                 title = 'Mapa com todas as casas, dividido por casas abaixo e acima do preço', size= 'previsao', size_max= 15, 
                                 template = 'plotly_dark',
                                 color_discrete_sequence=['#AEFF02', '#031147'],
                                 zoom=9)
mapa_final.update_layout(mapbox_style = 'carto-darkmatter')
mapa_final.update_layout(height = 700, width = 750, margin = {'r':0, 't':45, 'l':0, 'b':0})
mapa_final.show() 

data_definitivo = data_definitivo.rename(columns={100 : 'id',
                        101 : 'data',
                        102 : 'price',
                        103 : 'previsao',
                        104 : 'valor_casa'
                        })

abaixo = data_definitivo[data_definitivo.valor_casa == 'Abaixo']

(abaixo['previsao'] - abaixo['price']).sum() 

data_definitivo.dtypes

data_definitivo['price'] = data_definitivo['price'].astype(float)
data_definitivo['previsao'] = data_definitivo['previsao'].astype(float)

data_zipcode.head(50)

data1 = data
data1_x_kc = data.iloc[:, 3:18]
data1_y_kc = data.iloc[:, 2].values

# data1_x_kc.loc[0,'sqft_living'] = 1950
# data1_x_kc.loc[0,'bathrooms'] = 3
# data1_x_kc.loc[0,'condition'] = 4
# data1_x_kc.loc[0,'grade'] = 8
# data1_x_kc.loc[0,'zipcode'] = '98115'
# data1_x_kc.loc[0,'bedrooms'] = 5

data1_y_kc[0]
data1_x_kc = data1_x_kc.values

label_encoder_zipcode = LabelEncoder()
label_encoder_Renovated = LabelEncoder()
label_encoder_Basement = LabelEncoder()
label_encoder_mes_ano = LabelEncoder()
label_encoder_sazonal = LabelEncoder()
label_encoder_waterfront = LabelEncoder()

data1_x_kc[:,9] = label_encoder_zipcode.fit_transform(data1_x_kc[:,9])
data1_x_kc[:,10] = label_encoder_Basement.fit_transform(data1_x_kc[:,10])
data1_x_kc[:,11] = label_encoder_waterfront.fit_transform(data1_x_kc[:,11])
data1_x_kc[:,12] = label_encoder_Renovated.fit_transform(data1_x_kc[:,12])
data1_x_kc[:,13] = label_encoder_mes_ano.fit_transform(data1_x_kc[:,13])
data1_x_kc[:,14] = label_encoder_sazonal.fit_transform(data1_x_kc[:,14])

OneHotEncoder_hr = ColumnTransformer(transformers=[('Onehot', OneHotEncoder(), [9,10,11,12,13,14])], remainder = 'passthrough')
data1_x_kc = OneHotEncoder_hr.fit_transform(data1_x_kc)
data1_x_kc = data1_x_kc.toarray()

scaler_x_kc = StandardScaler()
data1_x_kc = scaler_x_kc.fit_transform(data1_x_kc)
scaler_y_kc = StandardScaler()
data1_y_kc = scaler_y_kc.fit_transform(data1_y_kc.reshape(-1,1))

novo_registro = data1_x_kc[0].reshape(1,-1)
test =  regressor_rna_kc.predict(novo_registro)
test = test.reshape(1,-1)
data1_y_kc = scaler_y_kc.inverse_transform(data1_y_kc)
test = scaler_y_kc.inverse_transform(test)
print(y_kc_scaled[0])
print(test)

