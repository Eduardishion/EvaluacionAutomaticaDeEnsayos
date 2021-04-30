from sklearn.neighbors.nearest_centroid import NearestCentroid

class clasificadorNCC():

	def entrenarClasificadorNCC(self, train_data_featuresTmp, train, train_data_features_test):

		#clf = NearestCentroid()
		clf = NearestCentroid(metric='euclidean', shrink_threshold=None)

		objPredicador = clf.fit(train_data_featuresTmp, train["sentiment"])

		result = objPredicador.predict(train_data_features_test)
		return result
