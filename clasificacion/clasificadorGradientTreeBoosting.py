from sklearn.ensemble import GradientBoostingClassifier


class clasificadorGTB():

    def entrenarClasificadorGTB(self, train_data_featuresTmp, train, train_data_features_test):

        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

        objPredicador = clf.fit(train_data_featuresTmp, train["sentiment"])

        result = objPredicador.predict(train_data_features_test)

        return result


