import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def data_report(df):
    '''Esta funcion describe los campos de un dataframe de pandas de forma bastante clara, crack'''
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T

#Estractor de unidades
def verificador_unidades(strings):
    unidades = set()
    for val in strings:
        words = val.split()
        for i in range(len(words) - 1):
            unit = words[i + 1]
            if unit in unidades:
                continue
            else:
                unidades.add(unit)
    return unidades

# Aplicar One_Hot_Encoder a Train y a Test
def apply_onehot_encoder(train:pd.DataFrame, columns_to_encode:list, test:pd.DataFrame=None):
    
    # Resetear índices para evitar desalineación
    train = train.reset_index(drop=True)
    
    # Crear el OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Ajustar y transformar las columnas seleccionadas
    transformed_data = encoder.fit_transform(train[columns_to_encode])

    # Crear un DataFrame con las columnas transformadas
    transformed_df = pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out(columns_to_encode))
    
    # Concatenar con el DataFrame original excluyendo las columnas transformadas
    df_concatenated = pd.concat([train.drop(columns_to_encode, axis=1), transformed_df], axis=1)

    # Si se proporciona un segundo DataFrame, aplicar la misma transformación
    if test is not None:
        test = test.reset_index(drop=True)
        transformed_data_to_transform = encoder.transform(test[columns_to_encode])
        transformed_df_to_transform = pd.DataFrame(transformed_data_to_transform, columns=encoder.get_feature_names_out(columns_to_encode))
        df_to_transform_concatenated = pd.concat([test.drop(columns_to_encode, axis=1), transformed_df_to_transform], axis=1)
        return df_concatenated, df_to_transform_concatenated

    return df_concatenated

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def graficar_histograma_y_evaluar_normalidad_visual(df):
    
    columnas_numericas = df.select_dtypes(include=[np.number]).columns

    #Configurar el tamaño de la columna
    num_col = len(columnas_numericas)
    plt.figure(figsize=(15, 5*num_col))

    #Iterar sobre las columnas numericas
    for i, columna in enumerate(columnas_numericas, 1):
        plt.subplot(num_col, 1, i)
        data = df[columna].dropna() #Excluir los NAN de la prueba de normalidad

        #Graficar Histograma y curva Normal
        sns.histplot(data, kde=True, stat='density')
        mu, std = norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth = 2)

        #Interpretacion visual
        if p.max() > 0.1: #Un umbral de ejemplo que podrias ajustar
            plt.title(f'{columna} - Distribucion visualmente cercana a la normal')
        else:
            plt.title(f'{columna} - Distribucion visualmente no normal')

    #plt.tigth_layout()
    plt.show()