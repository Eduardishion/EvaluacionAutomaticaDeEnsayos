#hacemos uso de metodo recien creado 

#impartacion de librerias necesarias

import pandas as pd 
from metodosDeTratamientoDeTextos import *


#creacion de un objeto de la clase preprocesamiento  
objPreprocesamiento = preposesamiento()

#lectura del archivo y configuracion para los delimitadores el cual contiene 25000 registros con 3 columnas 
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

#pasamos como parametros la primera fila del archivo el campo review para preprocesar el texto 
muestraLimpiada = objPreprocesamiento.review_to_words2(train["review"][0])

#ahora solo imprimimos el texto preprocesado
print muestraLimpiada

