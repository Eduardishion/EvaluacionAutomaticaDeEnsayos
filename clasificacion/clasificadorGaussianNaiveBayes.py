from sklearn.naive_bayes import GaussianNB

class clasificadorGNB():

	def entrenarClasificadorGNB(self, train_data_featuresTmp, train, train_data_features_test):

		gnb = GaussianNB()

		objPredicador = gnb.fit(train_data_featuresTmp, train["sentiment"])

		result = objPredicador.predict(train_data_features_test)

		return result