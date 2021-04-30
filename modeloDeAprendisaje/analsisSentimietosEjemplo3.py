#hacemos uso de metodo recien creado 
#impartacion de librerias necesarias
#en el siguiente ejemplo se vamos a procesar todos los resumenes lo cual 
# ahcemos uso de for 

import pandas as pd 
from metodosDeTratamientoDeTextos import *


#creacion de un objeto de la clase preprocesamiento  
objPreprocesamiento = preposesamiento()

#lectura del archivo y configuracion para los delimitadores el cual contiene 25000 registros con 3 columnas 
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# obtenemos el tamano de las filas del archivo 
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews, crear el vector que guardara todos los archivos 
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list # Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 

for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    
    #clean_train_reviews.append( review_to_words( train["review"][i] ) )
    clean_train_reviews.append(objPreprocesamiento.review_to_words2(train["review"][i]))


#pasamos como parametros la primera fila del archivo el campo review para preprocesar el texto 
#muestraLimpiada = objPreprocesamiento.review_to_words2(train["review"][0])

#ahora solo imprimimos el texto preprocesado
#print muestraLimpiada

