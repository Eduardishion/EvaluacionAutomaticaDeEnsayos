
from sklearn.linear_model import SGDClassifier


class clasificadorSGD():

    def entrenarClasificadorSGD(self, train_data_featuresTmp, train, train_data_features_test):
                #clf = SGDClassifier(loss="hinge", penalty="l2")
        clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
                            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                            learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
                            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
                            verbose=0, warm_start=False)

        objPredicador = clf.fit(train_data_featuresTmp, train["sentiment"])

        result = objPredicador.predict(train_data_features_test)

        return result
