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
from clasificacion.clasificadorVotingClassifier import *


from ejemploDeGraficaionDeDatos import *
from bs4 import BeautifulSoup
from modeloDeAprendisaje.medidasDeevalaucion import *
import numpy as np
#----------------------------------------------------------------------------------------------------------------------
#librerias para ejmplos de libro
from sklearn.model_selection import train_test_split

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
        train = pd.read_csv("../dataSets/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3,nrows = 500)

        # datos a evaluar
        # Read the test data
        print ("Carga de archivo de los datos de prueva ...")
        test = pd.read_csv("../dataSets/testData.tsv", header=0, delimiter="\t", quoting=3)
        # Verify that there are 25,000 rows and 2 columns
        # print test.shape

        # para el sugundo ejemplo de prediccion

        #salida = pd.DataFrame ( data={ "id": train["id"],"sentiment": train["sentiment"] } )
        #salida.to_csv ( "testpruevas.csv", index=False, quoting=3 )



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
        train_data_features = objBoldaDePalabras.obtenerCaracteristicas(clean_train_reviewsTmp)

        print (train_data_features[0])

        # bolsa de palabras de los datos de pruevas a predecir
        print ("Obtencion de bolsa de palabras de datos de prueva a predecir...")
        train_data_features_test = objBoldaDePalabras.obtenerCaracteristicas(clean_test_reviews)

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

    def crearModeloDeAprendisaje3(self):
        print ("Ejemplo de cargas de datos...")
        #objetos clasee para pre-prosesar el texto y generacion de bolsa de palabras
        objPreprocesamiento = preposesamiento ( )
        objBoldaDePalabras = bolsaDePalabras ( )

        # objetos para usar los clasificadores a usar y hacer pruevas ramdom forest, maquina de soporte vectorial, redes neuronal, vesinos mas cercanos, algoritmo de votacion
        objClasificador  = clasificadorRandomForest ( )
        objClasificador2 = clasificadorSVM ( )
        objClasificador6 = clasificadorANN ()
        objClasificador3 = clasificadorKNN ( )
        objClasificador12= clasificadorVC()
        objMedidaEval = evaluacion ( )

        #para cargar los archivos usar la muestra de entrenamiento, los datos de pruevas y el archivo para comparar las respuestas obtenidas
        print ("Carga de archivo de los datos de entrenamiento...")
        #datos de entrenamiento
        datosDeEntrenamiento = pd.read_csv ( "../dataSets/ensayos/train_rel_2.tsv", header=0, delimiter="\t", quoting=3,nrows = 1600)
        # para mostrar campos de los datos de entranamiento
        #print datosDeEntrenamiento.keys()

        #paramostrar columna especifica
        #print datosDeEntrenamiento['Id']
        #print datosDeEntrenamiento['EssaySet']
        #print datosDeEntrenamiento['Score1']
        #print datosDeEntrenamiento['EssayText']

        #para mostrar el tamño de la muestra
        #print datosDeEntrenamiento['Id'].shape

        #para mostrar un numero especifico de la muestra
        #print datosDeEntrenamiento['Id'][:5]

        #para mostrar los tipos de datos de los campos
        #print type(datosDeEntrenamiento['Id'])



        #datos de pruevas
        print ("Cargar de archivo de los datos de pruevas...")
        datosDePrueva = pd.read_csv ( "../dataSets/ensayos/public_leaderboard_rel_2.tsv", header=0, delimiter="\t", quoting=3 ,nrows = 558)
        # para mostrar campos de los datos de pruevas print datosDePrueva.keys()


        #datos de solucion
        #solucionPublic_leaderboard = pd.read_csv ( "../dataSets/ensayos/public_leaderboard_solution.csv" ,nrows = 558)
        solucionPublic_leaderboard = pd.read_csv ( "../dataSets/ensayos/bag_of_words_benchmark.csv", nrows=558 )

        # para tomar la muestra de pruevas
        #muestraTest = pd.read_csv ( "../dataSets/ensayos/train_rel_2.tsv", header=0, delimiter="\t", quoting=3,nrows = 55)

        #ejmeplo para ver los nombre de las columnas del archivo de solucion
        columnasSolucion = solucionPublic_leaderboard.columns.values
        for  i in  range(0, len(columnasSolucion)):
            print ("===",columnasSolucion[i])

        #print (solucionPublic_leaderboard["id"][0])
        #print (solucionPublic_leaderboard["essay_set"][0])
        #print (solucionPublic_leaderboard["essay_weight"][0])
        #print (solucionPublic_leaderboard["essay_score"][0])
        #print (solucionPublic_leaderboard["Usage"][0])

        #para mostrar los datos del archivo de solucion
        for i in range(0,10):
            #print (solucionPublic_leaderboard["id"][i], solucionPublic_leaderboard["essay_set"][i],  solucionPublic_leaderboard["essay_weight"][i], solucionPublic_leaderboard["essay_score"][i],solucionPublic_leaderboard["Usage"][i] )
            print (solucionPublic_leaderboard["id"][i],  solucionPublic_leaderboard["essay_score"][i])
        #mostrae el tamaño de la muestra solucion
        print ("el tamaño de la muestra es: ", len(solucionPublic_leaderboard))

        #muestra = pd.DataFrame (data={"Id": muestraTest["Id"],"EssaySet": muestraTest["EssaySet"], "EssayText": muestraTest["EssayText"]} )
        #muestra.to_csv( "muestraTest.csv")

        #otros breves ejemplos
        """
            df = pd.read_csv ( "../dataSets/ensayos/train.tsv", header=0, delimiter="\t", quoting=3 )
            #df = pd.read_csv ( '../dataSets/ensayos/private_leaderboard.tsv' )
            tamMuestra = df.columns.values
            for i in range(0, len(tamMuestra)):
                print ("--",tamMuestra[i])

            print ("\n")
        """

        """
            columnasMuestra2 = df.columns.values

            for i in  range(0, len(columnasMuestra2)):
                print(">>>",columnasMuestra2[i])

            print ("1:",df["id"][0])
            print ("2:",df["essay_set"][0])
            print ("3:",df["essay_weight"][0])
            #print ("4:", df["essay_score"][0])
            #print ("5:", df["Usage"][0])

            print datosDeEntrenamiento["EssayText"].size
            print datosDePrueva["EssayText"].size
        """

        numIntanciasEntrenamiento = datosDeEntrenamiento["EssayText"].size
        numIntanciasPruevas =  datosDePrueva["EssayText"].size


        #print (datosDeEntrenamiento.columns.values)
        columnasdatosDeEntrenamiento = datosDeEntrenamiento.columns.values
        columnasdatosPruevas =  datosDePrueva.columns.values

        print ("columnas de datos de entrenamiento")
        for i in range(0,len(columnasdatosDeEntrenamiento)):
            print (i)
            print (columnasdatosDeEntrenamiento[i])
        print ("-------")
        for i in range(0,len(columnasdatosPruevas)):
            print (columnasdatosPruevas[i])
        print ("-------")
        print ("entrenamiento",datosDeEntrenamiento["EssayText"].size)
        print ("pruevas",datosDePrueva["EssayText"].size)
        #print (df["EssayText"].size)



        #hay dos metodos para preprocesar el texto uno con Streming y Lemmatizer
        #hya otro sin aplicaar  Streming y Lemmatizer
        print ("Preprosesamiento de los datos de entrenamiento... ")
        #datosDeEntrenamientoLimpios = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayos(numIntanciasEntrenamiento,datosDeEntrenamiento,columnasdatosDeEntrenamiento[4])
        datosDeEntrenamientoLimpios = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayosYStreming(numIntanciasEntrenamiento, datosDeEntrenamiento, columnasdatosDeEntrenamiento[4] )
        #for i in range(0, 10):
        #    print (datosDeEntrenamientoLimpios[i])

        print ("Preposesamiento de los datos de pruevas...")
        #datosDepruevaLimpios = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayos (numIntanciasPruevas, datosDePrueva, columnasdatosPruevas[2] )
        datosDepruevaLimpios = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayosYStreming(numIntanciasPruevas, datosDePrueva, columnasdatosPruevas[2] )
        #for i in range(0, 10):
        #    print (datosDepruevaLimpios[i])




        # metodos para obtener  bolsa de palabras de los datos de entrenamienro
        print ("Obtencion de bolsa de palabras de datos de entrenamiento...")
        caracteristicasDeEntrenamiento = objBoldaDePalabras.extraccionDeCaracteristicasBoW2(datosDeEntrenamientoLimpios)
        #caracteristicasDeEntrenamiento = objBoldaDePalabras.obtenerBolsaDePalabrasNgramas( datosDeEntrenamientoLimpios )
        #caracteristicasDeEntrenamiento = objBoldaDePalabras.obtenerBolsaDePalabras_Tf_idf( datosDeEntrenamientoLimpios )
        #caracteristicasDeEntrenamiento = objBoldaDePalabras.obtenerBolsaDePalabras_with_TfidfVectorizer(datosDeEntrenamientoLimpios )

        # print (caracteristicasDeEntrenamiento[0])
        #caracteristicasDeEntrenamiento = objBoldaDePalabras.obtenerBolsaDePalabrasMejoresCaracteristicas( datosDeEntrenamientoLimpios,datosDeEntrenamiento["Score1"])
        #caracteristicasDeEntrenamiento = objBoldaDePalabras.obtenerBolsaDePalabrasNgramas(datosDeEntrenamientoLimpios)

        #caracteristicasSelectasEntrenamiento = objBoldaDePalabras.seleccionDeCaracteristicas ( caracteristicasDeEntrenamiento,columnasdatosDeEntrenamiento["Score1"])

        #for i in range(0, 10):
        #    print (caracteristicasDeEntrenamiento[i])

        #frecuenciasDeEntrenamiento = objBoldaDePalabras.obtenerFrecuenciaDeTerminostf(caracteristicasDeEntrenamiento)
        #frecuenciasDeEntrenamiento = objBoldaDePalabras.obtenerFrecuenciaInversaDelDocumentos(caracteristicasDeEntrenamiento)

        #for i in range(0, 1):
        #    print (frecuenciasDeEntrenamiento[i])

        #newcaracteristicasDeEntrenamiento = objBoldaDePalabras.seleccionDeCaracteristicasRemovingFeaturesLowVariance (caracteristicasDeEntrenamiento)
        #newcaracteristicasDeEntrenamiento = objBoldaDePalabras.seleccionDeCaracteristicas (caracteristicasDeEntrenamiento,datosDeEntrenamiento,columnasdatosDeEntrenamiento[2])


        print ("Obtencion de bolsa de palabras de datos de pruevas...")
        caracteristicasDePruevas = objBoldaDePalabras.extraccionDeCaracteristicasBoW2(datosDepruevaLimpios)
        #caracteristicasDePruevas = objBoldaDePalabras.obtenerBolsaDePalabrasNgramas( datosDepruevaLimpios )
        #caracteristicasDePruevas = objBoldaDePalabras.obtenerBolsaDePalabras_Tf_idf( datosDepruevaLimpios )
        #caracteristicasDePruevas = objBoldaDePalabras.obtenerBolsaDePalabras_with_TfidfVectorizer( datosDepruevaLimpios )
        #newcaracteristicasDePruevas = objBoldaDePalabras.seleccionDeCaracteristicasRemovingFeaturesLowVariance (caracteristicasDePruevas )





        for i in range(0, 10):
            print (caracteristicasDePruevas[i])
        #frecuenciasDePruevas = objBoldaDePalabras.obtenerFrecuenciaDeTerminostf (caracteristicasDePruevas)

        #frecuenciasDePruevas = objBoldaDePalabras.obtenerFrecuenciaInversaDelDocumentos (caracteristicasDePruevas)


        #newcaracteristicasDePruevas = objBoldaDePalabras.seleccionDeCaracteristicasSelectKBest(caracteristicasDePruevas, datosDeEntrenamiento, columnasdatosDeEntrenamiento[2] )

        #print ("Entremaniento del modelo del clasificador....RandomForest...")
        print ("Entremaniento del modelo del clasificador....")


        #del otro eejmplo de archivos
        #resultTmp = objClasificador3.entrenarClasificadorKNN(train_data_features,train,train_data_features_test)
        #resultTmp = objClasificador3.entrenarClasificadorKNN(train_data_features,train,train_data_features_test)



        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas ramdom forest 0.771920132292 con Streming y Lemmatizer
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas ramdom forest 0.767647156801
        resultTmp = objClasificador.entrenarClasificadoryObtenerResultadosEnsayos ( caracteristicasDeEntrenamiento,
                                                                                    datosDeEntrenamiento,
                                                                                    caracteristicasDePruevas,
                                                                                    columnasdatosDeEntrenamiento[2])
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas red neuronal 0.68328081734 con Streming y Lemmatizer
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas red neuronal 0.658847550082
        #resultTmp = objClasificador6.entrenarClasificadorNNEnsayos(caracteristicasDeEntrenamiento,
        #                                                                          datosDeEntrenamiento,
        #                                                                          caracteristicasDePruevas,
        #                                                                          columnasdatosDeEntrenamiento[2])
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas SVM  0.726550208365 con Streming y Lemmatizer
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas SVM  0.694620514896
        #resultTmp = objClasificador2.entrenarClasificadorSVMEnsayos(caracteristicasDeEntrenamiento,
        #                                                           datosDeEntrenamiento,
        #                                                           caracteristicasDePruevas,
        #                                                           columnasdatosDeEntrenamiento[2])
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas CV 0.751757005508 con Streming y Lemmatizer
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas CV 0.733205111088
        #resultTmp = objClasificador12.entrenarClasificadorclasificadorVCEnsayos(caracteristicasDeEntrenamiento,
        #                                                           datosDeEntrenamiento,
        #                                                           caracteristicasDePruevas,
        #                                                           columnasdatosDeEntrenamiento[2])
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas KNN 0.581025573075 con Streming y Lemmatizer
        #---------------------------------------------------------------------------------------------------------------con el que se esta haciendo pruevas KNN 0.52208491374
        #resultTmp = objClasificador3.entrenarClasificadorKNNEnsayos(caracteristicasDeEntrenamiento,
        #                                                           datosDeEntrenamiento,
        #                                                           caracteristicasDePruevas,
        #                                                           columnasdatosDeEntrenamiento[2])


        print ("Guardando resultados de la prediccion del modelo de clasificacion...")
        #output = pd.DataFrame ( data={"Id": datosDePrueva["Id"],"id_solu": solucionPublic_leaderboard["id"],"EssaySet":datosDePrueva["EssaySet"],"EssaySet_solu":solucionPublic_leaderboard["essay_set"], "Score_Clf": resultTmp, "Score_Eval": solucionPublic_leaderboard["essay_score"]} )
        output = pd.DataFrame ( data={"Id": datosDePrueva["Id"], "id_solu": solucionPublic_leaderboard["id"],
                                     "Score_Clf": resultTmp,
                                    "Score_Eval": solucionPublic_leaderboard["essay_score"]} )

        #output = pd.DataFrame ( data={ "Score1": resultTmp} )
        #essay_score

        #print ("Resultados de la prediccion guardandose en el archivo de resultados...")
        # Use pandas to write the comma-separated output file
        output.to_csv ( "Bag_of_Words_model_ensayos.csv", index=False, quoting=3 )

        # para evaluar el modelo de clasidicacion con metrica de evaluacion
        porcentajeEvalaucion  = objMedidaEval.metricaEvaluacionCohensKappa ( solucionPublic_leaderboard["essay_score"], resultTmp )
        print ("porcentaje de evaluacion del modelo ")
        print (porcentajeEvalaucion)

        porcentajeEvalaucion2 = confusion_matrix (solucionPublic_leaderboard["essay_score"], resultTmp )
        print ("porcentaje de evaluacion del modelo 2")
        print (porcentajeEvalaucion2)

        print ("porcentaje de evaluacion del modelo 3")
        print  (np.mean ( resultTmp == solucionPublic_leaderboard["essay_score"] ))

        #porcentajeEvalaucion3 = metrics.precision_recall_fscore_support(solucionPublic_leaderboard["essay_score"], resultTmp )
        #print ("porcentaje de evaluacion del modelo 3")
        #print (porcentajeEvalaucion3)

        #porcentajeEvalaucion4 =metrics.precision_score(solucionPublic_leaderboard["essay_score"], resultTmp)
        #print ("porcentaje de evaluacion del modelo 4")
        #print (porcentajeEvalaucion4)

        #porcentajeEvalaucion5 =metrics.recall_score(solucionPublic_leaderboard["essay_score"], resultTmp)
        #print ("porcentaje de evaluacion del modelo 5")
        #print (porcentajeEvalaucion5)

        #y_true = [2, 0, 2, 2, 0, 1]
        #y_pred = [2, 0, 2, 2, 0, 1]
        #k = cohen_kappa_score ( y_true, y_pred )
        #print (k)

        # estas pruevas se han hecho con el set 1 del corpus tanto el los datos como de entrenamiento solo del set1
        # el clasificacor que se esta usando fue la maquina de soporte vectorial
        #1 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 60 caracteristicas obtuve  con la evalaucion kappa 0.726550208365 como resurlado
        #2 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Ngramas (uni-gramas, bi-gramas, tri-gramas juntos ) a 100 caracteristicas obtuve  con la evalaucion kappa 0.700907717345 como resurlado+ç


        # estas pruevas se han hecho con el set 1 del corpus tanto el los datos como de entrenamiento solo del set1
        # el clasificacor que se esta usando fue la red neuronal
        # 1 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 60 caracteristicas obtuve  con la evalaucion kappa 0.682901949434 como resurlado
        # 7 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 80 caracteristicas obtuve  con la evalaucion kappa 0.474514169809 como resurlado
        # 8 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 30 caracteristicas obtuve  con la evalaucion kappa 0.568663627101 como resurlado
        # 9 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 100 caracteristicas obtuve  con la evalaucion kappa 0.0196856944313 como resurlado
        # 10 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 10 caracteristicas obtuve  con la evalaucion kappa 0.684916856551 como resurlado
        # 11 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 20 caracteristicas obtuve  con la evalaucion kappa 0.688862746965 como resurlado
        # 12 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras solamente a 50 caracteristicas obtuve  con la evalaucion kappa 0.375546853962 como resurlado



        # 2 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Ngramas (uni-gramas, bi-gramas, tri-gramas juntos ) a 100 caracteristicas obtuve  con la evalaucion kappa 0.461705172303 como resurlado+ç
        # 2 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Ngramas (uni-gramas, bi-gramas, tri-gramas juntos ) a 80 caracteristicas obtuve  con la evalaucion kappa 0.338966455122 como resurlado+ç
        # 3 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Ngramas (uni-gramas, bi-gramas, tri-gramas juntos ) a 60 caracteristicas obtuve  con la evalaucion kappa 0.477445535931 como resurlado+ç
        # 4 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Ngramas (uni-gramas, bi-gramas, tri-gramas juntos ) a 30 caracteristicas obtuve  con la evalaucion kappa 0.657060508553 como resurlado+ç
        # 5 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Ngramas (uni-gramas, bi-gramas, tri-gramas juntos ) a 20 caracteristicas obtuve  con la evalaucion kappa 0.666457542583 como resurlado+ç
        # 6 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Ngramas (uni-gramas, bi-gramas, tri-gramas juntos ) a 10 caracteristicas obtuve  con la evalaucion kappa 0.581104049193 como resurlado+ç




        # 13 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 60 caracteristicas obtuve  con la evalaucion kappa 0.574437715809 como resurlado+ç
        # 14 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 10 caracteristicas obtuve  con la evalaucion kappa 0.662320247637 como resurlado+ç
        # 15 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 30 caracteristicas obtuve  con la evalaucion kappa 0.0 como resurlado+ç
        # 16 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 40 caracteristicas obtuve  con la evalaucion kappa 0.0 como resurlado+ç
        # 17 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 50 caracteristicas obtuve  con la evalaucion kappa 0.51242472828 como resurlado+ç
        # 18 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 80 caracteristicas obtuve  con la evalaucion kappa 0.494046885103 como resurlado+ç
        # 19 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 100 caracteristicas obtuve  con la evalaucion kappa 0.166159442211 como resurlado+ç
        # 20 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con Tf_idf a 120 caracteristicas obtuve  con la evalaucion kappa 0.075668364365 como resurlado+ç


        #uso del algoritmo ramdom forest
        # 21 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  con la evalaucion kappa 0.759426441237 como resurlado+ç
        # 22 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  con la evalaucion kappa 0.77133416885 como resurlado+ç
        # 23 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 10 caracteristicas obtuve  con la evalaucion kappa 0.576949252952 como resurlado+ç
        # 24 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  con la evalaucion kappa 0.769741942629 como resurlado+ç
        # 25 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  con la evalaucion kappa 0.768458830938 como resurlado+ç
        # 26 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  con la evalaucion kappa 0.772260953773 como resurlado+ç
        # 27 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  con la evalaucion kappa 0.7527091943 como resurlado+ç
        # 34 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  acplicacndo seleccion de caracteristicas al los datos de entrenamiento caracteristicas obtuve con la evalaucion kappa 0.773976054745 como resurlado+ç
        # 35 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  acplicacndo seleccion de caracteristicas al los datos de entrenamiento caracteristicas obtuve con la evalaucion kappa 0.765803034695 como resurlado+ç
        # 36 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  acplicacndo seleccion de caracteristicas al los datos de entrenamiento caracteristicas obtuve con la evalaucion kappa 0.769634617282 como resurlado+ç
        # 37 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  acplicacndo seleccion de caracteristicas al los datos de entrenamiento caracteristicas obtuve con la evalaucion kappa 0.779058332136 como resurlado+ç



        #uso de algortimo voting y uso de tres clasificadores para que voten SVM,ANN,RF
        # 28 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 caracteristicas obtuve  con la evalaucion kappa 0.753743827993 como resurlado+ç
        # 29 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 100 caracteristicas obtuve  con la evalaucion kappa 0.176429329623 como resurlado+ç
        # 30 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 20 caracteristicas obtuve  con la evalaucion kappa 0.730309217692 como resurlado+ç
        # 31 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 acplicacndo seleccion de caracteristicas al los datos de entrenamiento caracteristicas obtuve  con la evalaucion kappa 0.760816082473 como resurlado+ç
        # 32 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 acplicacndo seleccion de caracteristicas al los datos de entrenamiento caracteristicas obtuve  con la evalaucion kappa 0.761868768151 como resurlado+ç
        # 33 pruevas que he hecho  limpieza de los datos con uso streaming y lemenizzacion con conteo de palabras a 60 acplicacndo seleccion de caracteristicas al los datos de entrenamiento caracteristicas obtuve  con la evalaucion kappa 0.756766476202 como resurlado+ç

    def crearModeloDeAprendisaje4(self):

        objPreprocesamiento = preposesamiento()
        objBoldaDePalabras = bolsaDePalabras()

        # objetos para usar los clasificadores a usar y hacer pruevas ramdom forest, maquina de soporte vectorial, redes neuronal, vesinos mas cercanos, algoritmo de votacion
        objClasificador = clasificadorRandomForest()
        objClasificador2 = clasificadorSVM()
        objClasificador6 = clasificadorANN()
        objClasificador3 = clasificadorKNN()
        objClasificador12 = clasificadorVC()
        objMedidaEval = evaluacion()

        print ("Carga de archivo de los datos de entrenamiento y pruevas...")
        # datos de entrenamiento
        datosDeEntrenamiento = pd.read_csv("../dataSets/ensayos/train_rel_2.tsv", header=0, delimiter="\t", quoting=3,nrows=1600)#1600 son los ensayos del set 1
        datosDePrueva = pd.read_csv("../dataSets/ensayos/public_leaderboard_rel_2.tsv", header=0, delimiter="\t",quoting=3, nrows=558)
        print datosDePrueva.shape


        #datos solucion
        solucionPublic_leaderboard = pd.read_csv("../dataSets/ensayos/bag_of_words_benchmark.csv", nrows=558)
        print solucionPublic_leaderboard.shape


        # para mostrar campos de los datos de entranamiento
        # print datosDeEntrenamiento.keys()

        # paramostrar columna especifica
        # print datosDeEntrenamiento['Id']
        # print datosDeEntrenamiento['EssaySet']


        # print datosDeEntrenamiento['Score1']    y
        # print datosDeEntrenamiento['EssayText'] x

        #ejmeplo de uso de la funcion
        #X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],random_state=0)
        #carga de los datos de pruevas

        #para aabar el tamaño de la muestra de datos de entrenamiento y de pruevas
        totaldeTextosEntrenamiento = datosDeEntrenamiento["EssayText"].size
        totaldeTextosDePrueva =  datosDePrueva["EssayText"].size

        #para saber y asignar los datos de las columnas
        columnasdatosDeEntrenamiento = datosDeEntrenamiento.columns.values
        columnasdatosPruevas = datosDePrueva.columns.values

        #total de la muestra y las columnas de la muestra
        print totaldeTextosEntrenamiento
        print columnasdatosDeEntrenamiento
        print totaldeTextosDePrueva
        print columnasdatosPruevas

        print ("Limpieza de los datos de entrenamiento y pruevas...")
        #X_train, X_test, y_train, y_test = train_test_split(datosDeEntrenamiento['EssayText'], datosDeEntrenamiento['Score1'] , random_state=0)
        X_train = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayosYStreming2(totaldeTextosEntrenamiento, datosDeEntrenamiento, columnasdatosDeEntrenamiento[4])
        X_test  = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayosYStreming2(totaldeTextosDePrueva, datosDePrueva, columnasdatosPruevas[2])

        #X_train = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayos(totaldeTextosEntrenamiento, datosDeEntrenamiento, columnasdatosDeEntrenamiento[4])
        #X_test = objPreprocesamiento.metodoParaLimpiarNnumeroDetextosConMuestreoEnsayos(totaldeTextosDePrueva,datosDePrueva,columnasdatosPruevas[2])



        y_train = datosDeEntrenamiento[columnasdatosDeEntrenamiento[2]]
        y_test  = solucionPublic_leaderboard["essay_score"]  #solucion del test

        print ("Extraccion de caracteristicas de datos de entrenamiento y pruevas...")
        X_train_c = objBoldaDePalabras.extraccionDeCaracteristicasBoW2(X_train)
        X_test_c = objBoldaDePalabras.extraccionDeCaracteristicasBoW2(X_test)

        # mostrar el tamaño de las muestras
        print X_train_c.shape
        print X_test_c.shape


        # print X_train.keys()
        # print X_test.keys()

        # para mostrar el comportamineto de las muestras de del conjunto
        #fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        #plt.suptitle("iris_pairplot")

        #for i in range(3):
            #for j in range(3):
                #ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
                #ax[i, j].set_xticks(())
                #ax[i, j].set_yticks(())
                #if i == 2:
                #    ax[i, j].set_xlabel(iris['feature_names'][j])
                #if j == 0:
                #    ax[i, j].set_ylabel(iris['feature_names'][i + 1])
                # if j > i:
                #    ax[i, j].set_visible(False)







        print ("Entremaniento del modelo del clasificador....")

        y_pred = objClasificador.entrenarClasificadoryObtenerResultadosEnsayos3(X_train_c,
                                                                                X_test_c,
                                                                                y_train,
                                                                                y_test)


        print ("Guardando resultados de la prediccion del modelo de clasificacion...")
        # output = pd.DataFrame ( data={"Id": datosDePrueva["Id"],"id_solu": solucionPublic_leaderboard["id"],"EssaySet":datosDePrueva["EssaySet"],"EssaySet_solu":solucionPublic_leaderboard["essay_set"], "Score_Clf": resultTmp, "Score_Eval": solucionPublic_leaderboard["essay_score"]} )
        output = pd.DataFrame(data={"Id": datosDePrueva["Id"], "id_solu": solucionPublic_leaderboard["id"],
                                    "Score_Clf": y_pred,
                                    "Score_Eval": solucionPublic_leaderboard["essay_score"]})

        # output = pd.DataFrame ( data={ "Score1": resultTmp} )
        # essay_score

        # print ("Resultados de la prediccion guardandose en el archivo de resultados...")
        # Use pandas to write the comma-separated output file
        output.to_csv("Bag_of_Words_model_ensayos.csv", index=False, quoting=3)


        #-------------------------
        # para evaluar el modelo np.mean(y_pred == y_test)
        print "Evaluacion del modelo:>>>>>>>>>>>>>>>>>>>>>>>>"
        print np.mean(y_pred == y_test)



        # para evaluar el modelo de clasidicacion con metrica de evaluacion
        porcentajeEvalaucion = objMedidaEval.metricaEvaluacionCohensKappa(solucionPublic_leaderboard["essay_score"],y_pred)
        print ("porcentaje de evaluacion del modelo ")
        print (porcentajeEvalaucion)

        porcentajeEvalaucion2 = confusion_matrix(solucionPublic_leaderboard["essay_score"], y_pred)
        print ("porcentaje de evaluacion del modelo 2")
        print (porcentajeEvalaucion2)

        print ("porcentaje de evaluacion del modelo 3")
        print  (np.mean(y_pred == solucionPublic_leaderboard["essay_score"]))

















