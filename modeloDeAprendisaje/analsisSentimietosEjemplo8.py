# -*- coding: utf8 -*-
# hacemos uso de metodo recien creado
# impartacion de librerias necesarias
# en el siguiente ejemplo se vamos a procesar todos los resumenes lo cual
# ahcemos uso de for
# pip install lxml para tratamiendo de html en todas sus verciones
# pip install -U scikit-learn
# pip install virtualenv para instalar los entornos virtuales
# para activar el entorno virtual y poder instalar librerias independientes del entorno virtual
# se usa la siguiente sentencia por emplo virtualenv <DIR> dir es el directorio donde activar el entorno  virtualenv C:\Users\USUARIO\Desktop\dev
# para activar el entorno virtual se hace uso de la siguiente sentencia source <DIR>/bin/activate   ejemplo    source C:\Users\USUARIO\Desktop\dev\bin\activate
# pip install -U scikit-learn necesario para hacer la borsa de palabras y generar un vocabulario de los archivos

######################################################################3Creating Features from a Bag of Words (Using scikit-learn)
######################################################################creacion de las caracteristicas de la bolsa de palabras
import pandas as pd
from bs4 import BeautifulSoup
from preProcesamiento.metodosDeTratamientoDeTextos import *
from vectorizacionDocumentos.metodosParaCreacionBolsaDePalabras import *
from clasificacion.clasificadorRandomForest import *
from clasificacion.clasificadorSupportVectorMachine import *
from clasificacion.clasificadorkVecinosMasCercanos import *
from clasificacion.clasificadorRegresionLineal import *
from clasificacion.clasificadorArbolesDeDecision import *
from clasificacion.clasificadorNeuralNetwork import *
from clasificacion.clasificadorStochasticGradientDescent import *
from clasificacion.clasificadorNearestCentroidClassifier import *
from clasificacion.clasificadorGaussianNaiveBayes import *
from clasificacion.clasificadorGradientTreeBoosting import *
from ejemploDeGraficaionDeDatos import *
from bs4 import BeautifulSoup

class generadorDeModeloDeAprendisaje:
    def __init__(self):
        pass

    def funDev(self):
        print ("creando bolsa de palabras desde inicio sin librerias...")
        #  frases a evaluar y que usaremos como dereferencia que funciona

        frase1 = "The cat sat on the hat"

        frase2 = "The dog ate the cat and the hat"

        objEjemplo = graficador()

        #objEjemplo.funGrafica()
        objEjemplo.funGrafica2()








    def crearModeloAprendisaje2(self):
        objPreprocesamiento = preposesamiento ()

        print ("Carga de archivo de los datos de entrenamiento...")
        #muestraDeEntrenamiento = pd.read_csv ("../dataSets/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
        #al usar header NOne quita las cabeseras de los archivos y asigna un numero consecutivo en las columnas
        muestraDeEntrenamiento = pd.read_csv("../dataSets/labeledTrainData.tsv",header = 0, delimiter="\t", quoting=3)

        #para mostrar las cebezeras del archivo csv
        #print muestraDeEntrenamiento.columns.values
        #listaDeCabezeras = muestraDeEntrenamiento.columns.values
        #for i in range(0, len(listaDeCabezeras)):
        #    print  "cabezera: "+listaDeCabezeras[i]


        #for i in range(0, 5):
        #    print "text: "+ muestraDeEntrenamiento["review"][i]

        p = "<a class='button-auto-width  564654 competition-rules__accept'>I Understand  5468798 and  sdfsd sdf Accept 21321545</a>"

        print ("sin limpieza")
        #print muestraDeEntrenamiento["review"][0]
        print (p)
        print ("con limpieza")

        #texto = muestraDeEntrenamiento["review"][0]

        pal = BeautifulSoup(p,"lxml").get_text()
        print (pal)

        print ("sin caracteres que no sean A - z")
        solo_letras = re.sub ( "[^a-zA-Z]", " ", pal ) # para remover cualquier otro caracter que no se an letras

        print (solo_letras)

        listapal = solo_letras.split()


        #para dividir la frase en una lista de palabras
        #listapal =  texto.split()
        for i in  range(0 , len(listapal)):
            print (listapal[i])
        #para vel saber cuantas palabras tiene el texto
        #print len(texto)




        #para guardar en DstaFrame que es el matriz de los datros se puede guardar directamente como un archivo en csv
        #muestraDeEntrenamiento.to_csv("../dataSets/muestraDeentranientoNueva.csv")
        #para volver a cargar los datos del archivo anterior
        #muestraNueva = pd.read_csv("../dataSets/muestraDeentranientoNueva.csv")





    def crearModeloDeAprendisaje(self):
        # creacion de un objeto de la clase preprocesamiento
        objPreprocesamiento = preposesamiento()
        objBoldaDePalabras = bolsaDePalabras()
        objClasificador = clasificadorRandomForest()
        objClasificador2 = clasificadorSVM()
        objClasificador3 = clasificadorKNN()
        objClasificador4 = clasificadorLR()
        objClasificador5 = clasificadorADD()
        objClasificador6 = clasificadorANN()
        objClasificador7 = clasificadorSGD()
        objClasificador8 = clasificadorNCC()
        objClasificador9 = clasificadorGNB()  # pendiente necesita mas memoria
        objClasificador10 = clasificadorGTB()  # pendiente necesita mas memoria

        # lista para guardar datos de entrenamiento
        clean_train_reviewsTmp = []
        # lista para guardar datos de pruevas a predecir
        clean_test_reviews = []

        # lectura del archivo y configuracion para los delimitadores el cual contiene 25000 registros con 3 columnas
        # funcion de pandas read_csv para leer archivos en csv

        # la sentencia pd.read carga el archivo separado por comas o por tabulacion y genera una matrix
        # sin cabeceras se asgna 0 en header, el cual se asigna un numero consecutivo de en la cabecera

        # ejemplo de limpiesa de texto HTML
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


        # Datos de entrnamiento
        print ("Carga de archivo de los datos de entrenamiento...")
        train = pd.read_csv("../dataSets/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

        # datos a evaluar
        # Read the test data
        print ("Carga de archivo de los datos de prueva ...")
        test = pd.read_csv("../dataSets/testData.tsv", header=0, delimiter="\t", quoting=3)
        # Verify that there are 25,000 rows and 2 columns
        # print test.shape

        # para el sugundo ejemplo de prediccion


        # obtenemos el tamano de las filas del archivo
        # Get the number of reviews based on the dataframe column size
        num_reviews = train["review"].size

        # Create an empty list and append the clean reviews one by one
        num_reviews_test = len(test["review"])

        # ejemplo 2limpieza de texto de texto de prueva del corpus
        # textoAlimpiar=train["review"][0]
        # textoHTML = BeautifulSoup(textoAlimpiar, 'html.parser')
        # textoLimpo = textoHTML.get_text()
        # print(textoLimpo)

        # objPreprocesamiento.metodoParaLimpiarNnumeroDetextos(num_reviews,train)
        # limpiesa de datos de entrenamiento

        print ("Preprosesamiento de los datos de entrenamiento... ")
        clean_train_reviewsTmp = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreo(num_reviews, train)

        print ("Presprocesamiento de los datos de pruevas a predecir...")
        # limpieza de datos de pruevas a predecir
        clean_test_reviews = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreo(num_reviews_test, test)

        # pasamos como parametros la primera fila del archivo el campo review para preprocesar el texto
        # muestraLimpiada = objPreprocesamiento.review_to_words2(train["review"][0])

        # ahora solo imprimimos el texto preprocesado
        # print muestraLimpiada

        # bolsa de palabras de los datos de entrenamienro
        print ("Obtencion de bolsa de palabras de datos de entrenamiento...")
        train_data_features = objBoldaDePalabras.obtenerBolsaDePalabras(clean_train_reviewsTmp)

        print (train_data_features[0])

        # bolsa de palabras de los datos de pruevas a predecir
        print ("Obtencion de bolsa de palabras de datos de prueva a predecir...")
        train_data_features_test = objBoldaDePalabras.obtenerBolsaDePalabras(clean_test_reviews)

        # aqui se entrenan las caracteristicas los datos de entrenamiento no los datos de pruevas
        print ("Entremaniento del modelo del clasificador....RandomForest...")
        # forestTmp = objClasificador.entrenarClasificador(train_data_features,train)
        # resultTmp = objClasificador10.entrenarClasificadorGTB(train_data_features,train,train_data_features_test)
        # resultTmp = objClasificador9.entrenarClasificadorGNB(train_data_features,train,train_data_features_test
        # resultTmp = objClasificador8.entrenarClasificadorNCC(train_data_features,train,train_data_features_test)
        # resultTmp = objClasificador7.entrenarClasificadorSGD(train_data_features,train,train_data_features_test)
        # resultTmp = objClasificador6.entrenarClasificadorNN(train_data_features,train,train_data_features_test)
        # resultTmp = objClasificador5.entrenarClasificadorADD(train_data_features,train,train_data_features_test)
        # resultTmp = objClasificador4.entrenarClasificadorLR(train_data_features,train,train_data_features_test)
        # resultTmp = objClasificador3.entrenarClasificadorKNN(train_data_features,train,train_data_features_test)
        # resultTmp = objClasificador2.entrenarClasificadorSVM(train_data_features,train,train_data_features_test)
        resultTmp = objClasificador.entrenarClasificadoryObtenerResultados(train_data_features, train,
                                                                           train_data_features_test)

        # prediccion de resultados de los datos de pruevas
        # tras el entrenamiento de los datos enternamiento con el modelo obtenido de la funcion anterior es
        # hora de evaluar el los datos de pruevas con el modelo obtenido
        # Use the random forest to make sentiment label predictions

        # print "Prediccion de resultados de de los datos de pruevas..."
        # result = forestTmp.predict(train_data_features_test)


        # se genera un dalida de los resultados para despues guardarlos en un archivo
        # Copy the results to a pandas dataframe with an "id" column and
        # a "sentiment" column
        print ("Guardando resultados de la prediccion del modelo de clasificacion...")
        output = pd.DataFrame(data={"id": test["id"], "sentiment": resultTmp})

        print ("Resultados de la prediccion guardandose en el archivo de resultados...")
        # Use pandas to write the comma-separated output file
        output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
