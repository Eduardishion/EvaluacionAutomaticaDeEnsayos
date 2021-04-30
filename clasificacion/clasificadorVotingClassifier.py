from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from itertools import product

class clasificadorVC():

    def entrenarClasificadorclasificadorVCEnsayos(self, train_data_featuresTmp, train, train_data_features_test,nombreColumna):

        clasificadorVSM = svm.SVC ( C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                                    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                                    max_iter=-1, probability=True, random_state=None, shrinking=True,
                                    tol=0.001, verbose=False )


        clasificadorANN = MLPClassifier ( activation='relu', alpha=1e-05, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=False,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
                              learning_rate_init=0.001, max_iter=200, momentum=0.9,
                              nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                              solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False )

        clasificadorRF = RandomForestClassifier ( n_estimators=100 )

        clasificadorVC = VotingClassifier ( estimators=[('dt',clasificadorVSM), ('knn', clasificadorANN), ('svc', clasificadorRF)], voting='soft',weights=[2, 1, 2] )


        clasificadorVSM = clasificadorVSM.fit ( train_data_featuresTmp , train[nombreColumna] )
        clasificadorANN =clasificadorANN.fit ( train_data_featuresTmp , train[nombreColumna]  )
        clasificadorRF =clasificadorRF.fit ( train_data_featuresTmp , train[nombreColumna] )
        clasificadorVC = clasificadorVC.fit ( train_data_featuresTmp , train[nombreColumna] )



        result = clasificadorVC.predict ( train_data_features_test )

        return  result

