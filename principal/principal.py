# -*- coding: utf8 -*-
# from modeloDeAprendisaje.analsisSentimietosEjemplo8 import generadorDeModeloDeAprendisaje
#from modeloDeAprendisaje.analsisSentimietosEjemplo8.generadorDeModeloDeAprendisaje
from modeloDeAprendisaje.analsisSentimietosEjemplo8 import generadorDeModeloDeAprendisaje
from modeloDeAprendisaje.analsisSentimietosEjemplo10 import generadorDeModeloDeAprendisaje
from correctorOrtografico import corrector
# from modeloDeAprendisaje.analsisSentimietosEjemplo8 import generadorDeModeloDeAprendisaje
							  
if __name__ == '__main__':
		"""
			metodo prinpal para crear el modelo de aprensiaje aun podemos mejorarlo
			creacion del onjeto creador del modelo SASS
		"""
		objetoModeloAprendisaje = generadorDeModeloDeAprendisaje()
		#objetoModeloAprendisaje.crearModeloDeAprendisaje()
		#objetoModeloAprendisaje.crearModeloAprendisaje2()
		#objetoModeloAprendisaje.crearModeloDeAprendisaje3()
		objetoModeloAprendisaje.crearModeloDeAprendisaje4()
		#objetoModeloAprendisaje.funDev()

		#texto= "A procedure is soposed to be showing you how to do this experiment. The experiment is missing a lot of things. For example, the first thing sayed in this procedure is determine the mass of four different samples. That should not be the first step in the procedure. It should be to gather the matirials, tell you how to set up the matirials . It should tell you what different liqueds that should be placed into each container and what to labble on the container. It should tell you the size of the container to. It should tell you what temeriture the room should be and how much of the solution should be placed into the container."


	#para instalar tensor flow
	#pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl para instalr tensorflow