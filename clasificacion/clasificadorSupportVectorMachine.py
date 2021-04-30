# -*- coding: utf8 -*-
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


class clasificadorSVM():


	def entrenarClasificadorSVMEnsayos(self, train_data_featuresTmp, train, train_data_features_test,nombreColumna):
		print ("Training the SVM...")

		# creamos el objeto creador de la maquina de soporte vectorial
		clasificadorVSM = svm.SVC()

		# clasificadorVSM =  svm.svc(kernel='linear', c=1, gamma=1)
		#clasificadorVSM = svm.SVC ( C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		#							decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
		#							max_iter=-1, probability=False, random_state=None, shrinking=True,
		#							tol=0.001, verbose=False )



		# mandmos a traer el la funcion de entrenamiento de la maquina de soporte
		objPredicador = clasificadorVSM.fit ( train_data_featuresTmp, train[nombreColumna] )

		# kelnel se puede cambiar por ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
		# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
		#    max_iter=-1, probability=False, random_state=None, shrinking=True,
		#    tol=0.001, verbose=False)
		# obtenemos un resultado de la clasificacion de cada documento

		#clasificadorVSM.score()

		result = objPredicador.predict ( train_data_features_test )

		return result

	def entrenarClasificadorSVM(self,train_data_featuresTmp,train,train_data_features_test):
		print ("Training the SVM...")

		#creamos el objeto creador de la maquina de soporte vectorial
		#clasificadorVSM = svm.SVC()
		#clasificadorVSM =  svm.svc(kernel='linear', c=1, gamma=1) 
		clasificadorVSM =  svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
		tol=0.001, verbose=False)

		#mandmos a traer el la funcion de entrenamiento de la maquina de soporte 
		objPredicador = clasificadorVSM.fit(train_data_featuresTmp, train["sentiment"] )
		
		#kelnel se puede cambiar por ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
		 # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		 #    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
		 #    max_iter=-1, probability=False, random_state=None, shrinking=True,
		 #    tol=0.001, verbose=False)
		#obtenemos un resultado de la clasificacion de cada documento
		result = objPredicador.predict(train_data_features_test)

		return result