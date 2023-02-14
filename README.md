# Proyecto 1
# <center> Prediccion de Precios en Bienes Raices - Mercado Australiano 
## Regresion Avanzada

![A screenshot of the PyCalc interface](house-prices.avif)
## Introduccion
  
### Integrates de Grupo:
* José Estensoro (josee906@gmail.com)
* Roger Patón (oviroger@gmail.com)

### Entendimiento del Negocio
  
Una empresa de vivienda con sede en EE. UU. llamada Surprise Housing ha decidido ingresar al mercado australiano. La empresa utiliza el análisis de datos para comprar casas a un precio inferior a sus valores reales y venderlas a un precio más alto. Con el mismo propósito, la empresa ha recopilado un conjunto de datos de la venta de casas en Australia. Los datos se proporcionan en el archivo CSV a continuación.

La compañía está buscando posibles propiedades para comprar e ingresar al mercado. Debe construir un modelo de regresión utilizando la regularización para predecir el valor real de las posibles propiedades y decidir si invertir en ellas o no.

La empresa quiere saber:

Qué variables son significativas para predecir el precio de una casa, y
Qué tan bien esas variables describen el precio de una casa.
Además, determine el valor óptimo de lambda para la regresión de Ridge y Lasso.

### Objetivo del Negocio:
  
Debe modelar el precio de las casas con las variables independientes disponibles. Luego, la gerencia utilizará este modelo para comprender cómo varían exactamente los precios con las variables. En consecuencia, pueden manipular la estrategia de la empresa y concentrarse en áreas que generarán altos rendimientos. Además, el modelo será una buena manera para que la gerencia entienda la dinámica de precios de un nuevo mercado.

  ### Prerrequisitos
  
  Python 3.x

  ### Librerias para el análisis
  ```python
  import os
  import numpy as np 
  import pandas as pd 
  from scipy import stats
  import matplotlib.pyplot as plt 
  import seaborn as sns
  from google.colab import drive
  from scipy.stats import kendalltau
  
  ```
  
  ### Librerias para el modelado
  
  ```python
  from sklearn.model_selection import KFold, cross_val_score
  from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
  from sklearn.metrics import mean_squared_error
  from sklearn.preprocessing import RobustScaler
  from sklearn.pipeline import make_pipeline
  from sklearn.metrics import accuracy_score
  from sklearn.linear_model import Lasso
  from sklearn.linear_model import Ridge
  from sklearn.metrics import r2_score
  from sklearn.linear_model import LassoCV
  from sklearn.model_selection import GridSearchCV
