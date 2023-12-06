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