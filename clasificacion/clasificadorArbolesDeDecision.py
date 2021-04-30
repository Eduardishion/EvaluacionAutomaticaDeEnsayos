from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


class clasificadorADD():

	def entrenarClasificadorADD(self,train_data_featuresTmp,train,train_data_features_test):

		ad = DecisionTreeClassifier(criterion='entropy', max_depth=5) # Creando el modelo
		
		objPredicador = ad.fit(train_data_featuresTmp, train["sentiment"] ) # Ajustando el modelo

		result = objPredicador.predict(train_data_features_test)

		return result
