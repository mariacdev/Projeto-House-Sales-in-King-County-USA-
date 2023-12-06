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

