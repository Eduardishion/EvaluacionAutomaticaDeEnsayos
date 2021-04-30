# -*- coding: utf8 -*-


from sklearn.externals import joblib


class persistenciaDelModelos ( ):
    def __init__(self):
        pass

    def guardarModeloDeAprendisaje(self, clf):
        joblib.dump(clf, 'persistenciaDelModeloDeClasificacion.pkl')

    def cargarModeloDeAprendisaje(self):
        clf = joblib.load('persistenciaDelModeloDeClasificacion.pkl')
        return clf
