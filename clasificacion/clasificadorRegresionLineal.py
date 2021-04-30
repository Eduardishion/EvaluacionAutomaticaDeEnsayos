# -*- coding: utf8 -*-
# importando el modelo de regresión lineal

from sklearn.linear_model import LinearRegression


class clasificadorLR():

	def entrenarClasificadorLR(self,train_data_featuresTmp,train,train_data_features_test):

		rl = LinearRegression() # Creando el modelo.

		objPredicador = rl.fit(train_data_featuresTmp, train["sentiment"] ) # ajustando el modelo

		result = objPredicador.predict(train_data_features_test)

		return result