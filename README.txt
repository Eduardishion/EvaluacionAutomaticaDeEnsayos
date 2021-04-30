System Short Answer Scoring   evaluación de escritura automática

Se uso un sistema operativo windows 7 en vercion de 32 bits
vercion de python 2.7 

para ver paquetes instalados con pip usar $> pip freeze

Podemos volcar ese contenido en un archivo de requisitos:
	pip freeze > requirements
Y usar ese archivo para instalar las dependencias necesarias de nuestro proyecto en cualquier ordenador o servidor:
	pip install -r requirements



Para hacer funcionar la herramienta se uso  los siguientes paquetes de python

el archivo requirements contiene lo neceario para poder correr el sistema con el comando 

pip install -r requirements 

sirve para instalar las librerias necesarias automaticamente 

donde encontrar los paquetes para instarlos localmente en la siguiente pagina 
http://www.lfd.uci.edu/~gohlke/pythonlibs/

paquetes nesesarios 
beautifulsoup4==4.5.3
pandas==0.19.2
numpy==1.12.0+mkl
scipy==0.18.1
scikit-learn==0.18.1
nltk==3.2.2

comandos para instalar paquetes de desde consola usando pip como instalador de paquetes

pip install numpy
pip install scipy 		este comando solo funcional para sistemas linuz y unix en windows recomensado usar la instalacion del paquete localmente 
pip install pandas
pip install scikit-learn
pip install beautifulsoup4
pip install nltk
despues de instalar nlkt
	import nltk
	e  instalar nltk.download() desde consola
	descargar stopwords
	descargar wordNet
	descargar punkt
	o desde el comando
	
	nltk.download('stopwords')
	nltk.download('wordnet')
	nltk.download('punkt')
	
pip install matplotlib
pip install networks 
pip install statsmodels

del paquete nltk cuando ya este instalado descargar el corpus de stopwords que es necesario para correr el sistema 


estos paquetes debe estar previamente instalado pip para instalarlo en el pront 
usar este comando pagina de referencia https://bootstrap.pypa.io/get-pip.py
https://packaging.python.org/installing/


$>python get-pip.py             este archivo esta en la la carpeta "libreriasInstalablesSinInternet" si no funciona usar get-pip2.py que esta en la misma carpetta 




donde poder encontrar los paquetes para instalar localmente sin internet se encuentran en la carpeta "libreriasInstalablesSinInternet" del proyecto


paquetes por default

appdirs==1.4.0
bzr==2.6b1
lxml==3.7.3
packaging==16.8
pyparsing==2.1.10
python-dateutil==2.6.0
pytz==2016.10
six==1.10.0
virtualenv==15.1.0



Estructura de los paquetes de la herramienta
sASS/                            Paquete superior
      __init__.py                Inicializa el paquete 
      clasificacion/             Subpaquete para 
              __init__.py
              wavread.py
              ...
      dataSets/                  Subpaquete para guardar los dataSets
              __init__.py
              ...
      modeloDeAprensisaje/       Subpaquete para que genera el modelo de aprendisaje supervisado
              __init__.py
              ...
      persistenciaDeModelo/	 Subpaquete para guardar el modelo 
            __init__.py
              ...
      preProsesamiento/		 Subpaquete para librerias de preprocesamiento del lenguaje natural
            __init__.py
              ...
      principal/		 Subpaquete que contiene el programa principal en cual se usa para ejecutar el programa
            __init__.py
	    principal.py	
              ...
      vectorizacionDocumentos/   Subpaquete para la vectorizacion de los textos 
            __init__.py
              ...

