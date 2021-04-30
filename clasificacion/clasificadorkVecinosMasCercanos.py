from sklearn.neighbors import KNeighborsClassifier

class clasificadorKNN():

	def entrenarClasificadorKNN(self,train_data_featuresTmp,train,train_data_features_test):

		#knn = KNeighborsClassifier() # Creando el modelo con 10 vecinos
		knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

		objPredicador = knn.fit(train_data_featuresTmp, train["sentiment"]) # Ajustando el modelo

		result = objPredicador.predict(train_data_features_test)

		return result

	def entrenarClasificadorKNNEnsayos(self, train_data_featuresTmp, train, train_data_features_test, nombreColumna):
		# knn = KNeighborsClassifier() # Creando el modelo con 10 vecinos
		knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
								   metric_params=None, n_jobs=1, n_neighbors=5, p=2,
								   weights='uniform')

		objPredicador = knn.fit(train_data_featuresTmp, train[nombreColumna])  # Ajustando el modelo

		result = objPredicador.predict(train_data_features_test)

		return result