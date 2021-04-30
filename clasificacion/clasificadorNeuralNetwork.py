from sklearn.neural_network import MLPClassifier

class clasificadorANN():

    def entrenarClasificadorNN(self,train_data_featuresTmp,train,train_data_features_test):

        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

       clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

       objPredicador = clf.fit(train_data_featuresTmp, train["sentiment"] )

       result = objPredicador.predict(train_data_features_test)

       return result


    def entrenarClasificadorNNEnsayos(self, train_data_featuresTmp, train, train_data_features_test,nombreColumna):
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

        clf = MLPClassifier ( activation='relu', alpha=1e-05, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=False,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
                              learning_rate_init=0.001, max_iter=200, momentum=0.9,
                              nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                              solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False )

        objPredicador = clf.fit ( train_data_featuresTmp, train[nombreColumna] )



        result = objPredicador.predict ( train_data_features_test )

        return result
