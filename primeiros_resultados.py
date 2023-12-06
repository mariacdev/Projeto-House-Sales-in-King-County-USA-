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


data = pd.read_csv('kc_definitivo.csv')

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
