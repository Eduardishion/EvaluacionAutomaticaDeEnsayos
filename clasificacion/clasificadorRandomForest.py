# -*- coding: utf8 -*-
from sklearn.ensemble import RandomForestClassifier
from persistenciaDeModelo.persistencia import *
from sklearn.model_selection import cross_val_score

# ------------------------------------ para seleccion de caracteristicas
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from seleccionCaracteristicas.boruta_py import BorutaPy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class clasificadorRandomForest ( ):
    def __init__(self):
        pass

    def entrenarClasificador(self, train_data_featuresTmp, train):
        print ("Training the random forest...")
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier ( n_estimators=100 )

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        forest = forest.fit ( train_data_featuresTmp, train["sentiment"] )

        return forest

    def entrenarClasificadoryObtenerResultados(self, train_data_featuresTmp, train, train_data_features_test):
        objPer = persistenciaDelModelos ( )

        print ("Training the random forest...")
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier ( n_estimators=100 )

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        # This may take a few minutes to run
        forest = forest.fit ( train_data_featuresTmp, train["sentiment"] )

        print ("Prediccion de resultados de de los datos de pruevas...")

        result = forest.predict ( train_data_features_test )

        # para guardar el modelo de clasificacion y volverlo a usar
        objPer.guardarModeloDeAprendisaje ( forest )

        # para evaluar el modelo de clasidicacion con metrica de evaluacion
        porcentajeEvalaucion = objMedidaEval.metricaEvaluacionCohensKappa ( train["sentiment"], result )
        print ("porcentaje de evaluacion del modelo ")
        print porcentajeEvalaucion

        return result

    def entrenarClasificadoryObtenerResultados2(self, train_data_featuresTmp, train, train_data_features_test):
        objPer = persistenciaDelModelos ( )
        # objMedidaEval = evaluacion ( )

        print ("Training the random forest...")
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier ( n_estimators=100 )

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        # This may take a few minutes to run
        forest = forest.fit ( train_data_featuresTmp, train["sentiment"] )

        print ("Prediccion de resultados de de los datos de pruevas...")

        result = forest.predict ( train_data_features_test )

        # para guardar el modelo de clasificacion y volverlo a usar
        objPer.guardarModeloDeAprendisaje ( forest )

        # para evaluar el modelo de clasidicacion con metrica de evaluacion
        ##porcentajeEvalaucion = objMedidaEval.metricaEvaluacionCohensKappa ( train["sentiment"], result )
        # print ("porcentaje de evaluacion del modelo ")
        # print porcentajeEvalaucion

        return result

    def entrenarClasificadoryObtenerResultadosEnsayos(self, train_data_featuresTmp,
                                                      train,
                                                      train_data_features_test,
                                                      nombreColumna):
        objPer = persistenciaDelModelos ( )

        print ("Training the random forest...")
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier ( n_estimators=100 )

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        # This may take a few minutes to run
        # x								#y
        forest = forest.fit ( train_data_featuresTmp, train[nombreColumna] )

        print ("Prediccion de resultados de de los datos de pruevas...")

        result = forest.predict ( train_data_features_test )

        # para guardar el modelo de clasificacion y volverlo a usar
        objPer.guardarModeloDeAprendisaje ( forest )

        # scores = cross_val_score ( forest,train_data_featuresTmp , train[nombreColumna] )
        # print ("Evaluacion",scores.mean ())
        print ("evaluacion", forest.score ( train_data_featuresTmp, train[nombreColumna] ))

        return result

    # metodo con selecion de caracteristicas
    def entrenarClasificadoryObtenerResultadosEnsayos2(self, train_data_featuresTmp,
                                                       train_data_features_test,
                                                       target,
                                                       y_test):
        select = SelectFromModel ( RandomForestClassifier ( n_estimators=100, random_state=42 ), threshold="mean" )

        # seleccion de caracteristicas para  X_train
        select.fit ( train_data_featuresTmp, target )
        X_train_l1 = select.transform ( train_data_featuresTmp )
        # print(train_data_featuresTmp.shape)
        print(X_train_l1.shape)

        # seleccion de caracteristicas para  X_test_c
        X_test_l1 = select.transform ( train_data_features_test )
        # print(train_data_features_test.shape)
        print(X_test_l1.shape)

        # print(train_data_featuresTmp.shape)
        # print(X_train_l1.shape)


        objPer = persistenciaDelModelos ( )

        print ("Training the random forest...")
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier ( n_estimators=100 )

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        # This may take a few minutes to run
        # x		#y
        forest = forest.fit ( X_train_l1, target )

        print ("Prediccion de resultados de de los datos de pruevas...")

        result = forest.predict ( X_test_l1 )

        # para guardar el modelo de clasificacion y volverlo a usar
        objPer.guardarModeloDeAprendisaje ( forest )

        # scores = cross_val_score ( forest,train_data_featuresTmp , train[nombreColumna] )
        # print ("Evaluacion",scores.mean ())
        print ("evaluacion", forest.score ( X_train_l1, target ))

        print "Evaluacion del modelo2:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        print forest.score ( X_test_l1, y_test )

        return result

    # sin seleccion de caracteristicas
    def entrenarClasificadoryObtenerResultadosEnsayos3(self, train_data_featuresTmp,
                                                       train_data_features_test,
                                                       target,
                                                       y_test):
        # 1
        # select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold= "mean")

        # 2
        # select = SelectPercentile(percentile=50)

        # 3
        #rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=7)
        #select = BorutaPy(rf, n_estimators = 'auto', verbose=2)

        #4
        select = SelectKBest(chi2, k=2) #k es el numero de caracteristicas a seleccionar



        # 1
        # seleccion de caracteristicas para  X_train
        # select.fit(train_data_featuresTmp, target)
        # X_train_l1 = select.transform(train_data_featuresTmp)
        # print(train_data_featuresTmp.shape)
        # print(X_train_l1.shape)

        # 2
        # select.fit(train_data_featuresTmp, target)
        # transform training set:
        # X_train_selected = select.transform(train_data_featuresTmp)

        # 3
        #entrenar al seleccionador de caracteristicas de la muestra de entrenamiento
        #select.fit ( train_data_featuresTmp, target )
        #select.support
        #selecciona las caracteristicas mas adecuadas de la muestra de entranamiento
        #X_train_selected = select.transform ( train_data_featuresTmp )
        #selecciona las caracteristicas dela muestra de pruevas
        #X_test_l1 = select.transform ( train_data_features_test )

        #4
        select.fit_transform (train_data_featuresTmp, target  )
        X_train_selected = select.transform ( train_data_featuresTmp )
        # selecciona las caracteristicas dela muestra de pruevas
        X_test_l1 = select.transform ( train_data_features_test )





        print(X_train_selected.shape)
        print(X_test_l1.shape)

        objPer = persistenciaDelModelos ( )

        print ("Training the random forest...")
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier ( n_estimators=100 )

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        # This may take a few minutes to run
        # x		#y
        forest = forest.fit ( X_train_selected, target )

        print ("Prediccion de resultados de de los datos de pruevas...")

        result = forest.predict ( X_test_l1 )

        # para guardar el modelo de clasificacion y volverlo a usar
        objPer.guardarModeloDeAprendisaje ( forest )

        # scores = cross_val_score ( forest,train_data_featuresTmp , train[nombreColumna] )
        # print ("Evaluacion",scores.mean ())
        print ("evaluacion", forest.score ( X_train_selected, target ))

        print "Evaluacion del modelo2:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        print forest.score ( X_test_l1, y_test )

        return result

    def entrenarClasificadosyResultados(self, train_data_featuresTmp, train, train_data_features_test):
        print ("Training the random forest...")
        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier ( n_estimators=100 )

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        # This may take a few minutes to run
        forest = forest.fit ( train_data_featuresTmp, train["sentiment"] )

        print ("Prediccion de resultados de de los datos de pruevas...")

        result = forest.predict ( train_data_features_test )

        return result
