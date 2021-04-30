# -*- coding: utf8 -*-
#Cargamos los datos de Iris desde Scikit-learn

#Graficamos

#Importamos las librerias necesarias
#primere ejemplo
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
#para segundo ejemplo
import mlpy
from sklearn.cluster import KMeans

#enemplo 3
from numpy import ones,argmin

class graficador:

    def funGrafica(self):
        # Cargamos los datos y graficamos

        datos = load_iris ( )
        caract = datos.data
        caract_names = datos.feature_names
        tar = datos.target

        # Graficamos los datos con colores distintos y tipo de marcas distintos

        for t, marca, c in zip ( xrange ( 3 ), ">ox", "rgb" ):
            plt.scatter ( caract[tar == t, 0], caract[tar == t, 1], marker=marca, c=c )

        plt.show ( )


    def funGrafica2(self):
        # Cargamos los datos y graficamos

        datos = load_iris ( )
        dat = datos.data
        caract_names = datos.feature_names
        tar = datos.target

        # Calculamos los cluster

        cls, means, steps = mlpy.kmeans ( dat, k=3, plus=True )

        # steps
        # Esta variable permite conocer los pasos que realizó el algoritmo para terminar

        # Construimos las gráficas correspondiente

        plt.subplot ( 2, 1, 1 )
        fig = plt.figure ( 1 )
        fig.suptitle ( "Ejemplo de k-medias", fontsize=15 )
        plot1 = plt.scatter ( dat[:, 0], dat[:, 1], c=cls, alpha=0.75 )
        # Agregamos las Medias a las gráficas

        plot2 = plt.scatter ( means[:, 0], means[:, 1], c=[1, 2, 3], s=128, marker='d' )
        # plt.show()

        # Calculamos lo mismo mediante la librería scikit-lean

        KM = KMeans ( init='random', n_clusters=5 ).fit ( dat )

        # Extraemos las medias

        L = KM.cluster_centers_

        # Extraemos los valores usados para los calculos

        Lab = KM.labels_

        # Generamos las gráfica

        plt.subplot ( 2, 1, 2 )
        fig1 = plt.figure ( 1 )
        fig.suptitle ( "Ejemplo de k-medias", fontsize=15 )
        plot3 = plt.scatter ( dat[:, 0], dat[:, 1], c=Lab, alpha=0.75 )

        # Agregamos las Medias a las gráficas

        plot4 = plt.scatter ( L[:, 0], L[:, 1], c=[1, 2, 3, 4, 5], s=128, marker='d' )

        # Mostramos la gráfica con los dos calculos

        plt.show ( )



        """ algortimo k-medias
            #Idea General del algoritmo
            1)Inicialización
                -Se eligen la cantidad de clusters k
                -Se elige aleatóriamente k posiciones desde los datos de entrada
                -Se indica que el centro de los clusters tienen la posición de la media de los datos

            2)Aprendizale
            -Se repiten los pasos:
                Para cada punto de los datos se hace:
                -Se calcula la distancia del punto al centro         del Cluster
                -Se aglomera los puntos con el centro del clu        ster más cercano
                Para cada Cluster
                -Se cambia la poción del cluster al centro al        centroide de los puntos agrupados o aglomera        dos
                -Se hacen cambios en el centro del Cluster ha        sta que ya no hay variación en el cambio de         posición
            3)Tratamiento
                Para cada punto de prueba
                    -Se calcula la distancia a cada cluster
                    -Se asigna el punto al cluster con el cual ti        ene la distancia menor
        """


    def funGrafica3(self):
        # Calculo de distancia


        """
        dis = ones ( (1, self.nData) ) * sum ( (data - self.centre[0, :]) ** 2, axis=1 )

        for i in range ( self.k - 1 ):
            dis = append ( dis, ones ( (1, self.nData) ) * sum ( (data - self.centres[j - 1, :]) ** 2, axis=1 ),
                           axis=0 )

        # Se identifican los clusters
        cluster = dis.argmin ( axis=0 )
        cluster = traspose ( cluster * ones ( (1, self.nData) ) )

        # Se actualizan los centros
        for j in range ( self.k ):
            thisCluster = where ( cluster == j, 1, 0 )
            if sum ( thisCluster ) > 0:
                self.centres[j, :] = sum ( data * thisCluster, axis=0 ) / sum ( thisCluster )
        """





