#hacemos uso de metodo recien creado 
#impartacion de librerias necesarias
#en el siguiente ejemplo se vamos a procesar todos los resumenes lo cual 
# ahcemos uso de for 
#pip install lxml para tratamiendo de html en todas sus verciones 
#pip install -U scikit-learn 
#pip install virtualenv para instalar los entornos virtuales
#para activar el entorno virtual y poder instalar librerias independientes del entorno virtual 
#se usa la siguiente sentencia por emplo virtualenv <DIR> dir es el directorio donde activar el entorno  virtualenv C:\Users\USUARIO\Desktop\dev
#para activar el entorno virtual se hace uso de la siguiente sentencia source <DIR>/bin/activate   ejemplo    source C:\Users\USUARIO\Desktop\dev\bin\activate 
#pip install -U scikit-learn necesario para hacer la borsa de palabras y generar un vocabulario de los archivos 

######################################################################3Creating Features from a Bag of Words (Using scikit-learn)
######################################################################creacion de las caracteristicas de la bolsa de palabras 
import pandas as pd 
from bs4 import BeautifulSoup
from metodosDeTratamientoDeTextos import *
from metodosParaCreacionBolsaDePalabras import *
from clasificadorRandomForest import *

#creacion de un objeto de la clase preprocesamiento  
objPreprocesamiento = preposesamiento()
objBoldaDePalabras = bolsaDePalabras()
objClasificador = clasificadorRandomForest()
clean_train_reviewsTmp = []


#lectura del archivo y configuracion para los delimitadores el cual contiene 25000 registros con 3 columnas 
#funcion de pandas read_csv para leer archivos en csv

# la sentencia pd.read carga el archivo separado por comas o por tabulacion y genera una matrix 
#sin cabeceras se asgna 0 en header, el cual se asigna un numero consecutivo de en la cabecera 

#ejemplo de limpiesa de texto HTML
# html_doc = """
# 				<html>
# 				 <head>
# 				  <title>
# 				   The Dormouse's story
# 				  </title>
# 				 </head>
# 				 <body>
# 				  <p class="title">
# 				   <b>
# 				    The Dormouse's story
# 				   </b>
# 				  </p>
# 				  <p class="story">
# 				   Once upon a time there were three little sisters; and their names were
# 				   <a class="sister" href="http://example.com/elsie" id="link1">
# 				    Elsie
# 				   </a>
# 				   ,
# 				   <a class="sister" href="http://example.com/lacie" id="link2">
# 				    Lacie
# 				   </a>
# 				   and
# 				   <a class="sister" href="http://example.com/tillie" id="link2">
# 				    Tillie
# 				   </a>
# 				   ; and they lived at the bottom of a well.
# 				  </p>
# 				  <p class="story">
# 				   ...
# 				  </p>
# 				 </body>
# 				</html>
#             """
# textoHTML = BeautifulSoup(html_doc, 'html.parser')
# textoLimpio = textoLimpioDeEquiteasHTML.get_text()
# print(textoLim)



train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


# obtenemos el tamano de las filas del archivo 
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size


# ejemplo 2limpieza de texto de texto de prueva del corpus
# textoAlimpiar=train["review"][0]
# textoHTML = BeautifulSoup(textoAlimpiar, 'html.parser')
# textoLimpo = textoHTML.get_text()
# print(textoLimpo)

#objPreprocesamiento.metodoParaLimpiarNnumeroDetextos(num_reviews,train)
clean_train_reviewsTmp = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreo(num_reviews,train)

#pasamos como parametros la primera fila del archivo el campo review para preprocesar el texto 
#muestraLimpiada = objPreprocesamiento.review_to_words2(train["review"][0])

#ahora solo imprimimos el texto preprocesado
#print muestraLimpiada

train_data_features = objBoldaDePalabras.obtenerBolsaDePalabras(clean_train_reviewsTmp)



objClasificador.entrenarClasificador(train_data_features,train)

