#como instalar librerias por archivo teniendo ya instalado pip 
#pip install elpaquete.whl
#pip install pandas
#pip install -U numpy
#pip install pandas-0.19.2-cp27-cp27m-win32.whl
#pip install BeautifulSoup4
#pip install -U nltk



# Import the pandas package, then use the "read_csv" function to read
# the labeled training data

#importascion de libreria pandas para mejorar la lectura de archivos grandes 

#librerias qe se van usansando anotando todo lo que se pueda 
#para poder haber instalado pandas se descargo de pagina oficial 
#se uso pip para instalarlo localmente 
#recordar como instalar el pip para poder instalar mas paquetes 
#se a encontrado un repositorio de paquetes disponibles 
#el pandas instalado fue la vercion para 32 bits el de 64 dio error 


import pandas as pd 
from bs4 import BeautifulSoup # se utiliza para limpiar paracteres HTML el primer preproseso del texto
import re #libreria para manejo de expreciones regulares 
import nltk #que se utilizara para obtener palabras claves stopwods
from nltk.corpus import stopwords # Import the stop word list importa las stop words

# lectura del archivo separado por tabulacion con extencion tsv 
# la con configuracion de los respectivos delimitadores 
# header que inicio de la cabezera como 0
# delimiter que indica que es una tabulacion 
# quotin indica que  ignore las comillas 

# Reading the Data
# The necessary files can be downloaded from the Data page. The first file that you'll need is unlabeledTrainData.tsv, which contains 25,000 IMDB movie reviews, each with a positive or negative sentiment label.
# Next, read the tab-delimited file into Python. To do this, we can use the pandas package, introduced in the Titanic tutorial, which provides the read_csv function for easily reading and writing data files. If you haven't used pandas before, you may need to install it.

#lectura del archivo y configuracion para los delimitadores 
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


# Here, "header=0" indicates that the first line of the file contains column names, "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file.
# we make sure that we read 25,000 rows and 3 columns as follows:

# comandos para usarlos por consola y el primer comando  verificar la cantidad de registros y cuantas columnas 
#muestra el total de filas de los registros por el total de columnas
#train.shape
#train.columns.values #asigna columnas al archivo train
#muestra la primera fila del documento y el contenido de la columna review 
#print train["review"][0] imprime el primer registro el campo review que contiene el texto a preporcesar
#print train["review"][0]

# se necesita otra nueva libreria que inicial la limpiesa y se llama  beautifulsoup4-4.5.3-py2-none-any se debe tambien instalar
#pip install BeautifulSoup4 con el siguiente comando desde la siguiente teniendo ya instalado pip previamente 


# from the command line (NOT from within Python). Then, from within Python, load the package and use it to extract the text from a review:
# Import BeautifulSoup into your workspace
            

# Initialize the BeautifulSoup object on a single movie review
#se hace uso de beatifulSpup para limpiar el texto de etiquetas HTML
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison
##mostrar como se limpian los archivos ya limpiados de etiquetas HTML 

#print train["review"][0]
#print example1.get_text()


# Calling get_text() gives you the text of the review, without tags or markup. If you browse the BeautifulSoup documentation, you'll see that it's a very powerful library - more powerful than we need for this dataset. However, it is not considered a reliable practice to remove markup using regular expressions, so even for an application as simple as this, it's usually best to use a package like BeautifulSoup.
# Dealing with Punctuation, Numbers and Stopwords: NLTK and regular expressions
# When considering how to clean the text, we should think about the data problem we are trying to solve. For many problems, it makes sense to remove punctuation. On the other hand, in this case, we are tackling a sentiment analysis problem, and it is possible that "!!!" or ":-(" could carry sentiment, and should be treated as words. In this tutorial, for simplicity, we remove the punctuation altogether, but it is something you can play with on your own.
# Similarly, in this tutorial we will remove numbers, but there are other ways of dealing with them that make just as much sense. For example, we could treat them as words, or replace them all with a placeholder string such as "NUM".
# To remove punctuation and numbers, we will use a package for dealing with regular expressions, called re. The package comes built-in with Python; no need to install anything. For a detailed description of how regular expressions work, see the package documentation. Now, try the following:
# Use regular expressions to do a find-and-replace
#busca en el tesco letras que no sena ninguan otro caracter de a hasta z en minuscula 
#y tambien matusculas y otro caracter que no sea se conviente en un espacio en blanco 
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search

#se muestra el contenido ya formateado y cambiado las letras no exitentes a usar en espacios en balnco 
#tener en cuenta que el uso de esta expresion regular puede cambiar ya que para otro tipo de texto 
#se necesiten otras expreciones regulares para hacer una limpieza del texto  , medio ley que no es 
#recomendable el uso de expresiones regulares para eliminar etiquetas HTML, se recomienda en uso beutifulSoap

print letters_only

# A full overview of regular expressions is beyond the scope of this tutorial, but for now it is sufficient to know that [] indicates group membership and ^ means "not". In other words, the re.sub() statement above says, "Find anything that is NOT a lowercase letter (a-z) or an upper case letter (A-Z), and replace it with a space."
# We'll also convert our reviews to lower case and split them into individual words (called "tokenization" in NLP lingo):

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words

#aqui se van a buscar las palabras principales del idioma ingles
#uso de la libreia nlkt para obtener palabras mas comunes "stop words"
#sentencia para instalar corpus de nltk
#nltk.download()  # Download text data sets, including stop words descarga los distintos corpus de nltk
#print stopwords.words("english") #muestra las palabras mas principales del idioma ingles

# This will allow you to view the list of English-language stop words. To remove stop words from our movie review, do:

# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
# This looks at each word in our "words" list, and discards anything that is found in the list of stop words. After all of these steps, your review should now begin something like this:

print words
# Don't worry about the "u" before each word; it just indicates that Python is internally representing each word as a unicode string.
# There are many other things we could do to the data - For example, Porter Stemming and Lemmatizing (both available in NLTK) would allow us to treat "messages", "message", and "messaging" as the same word, which could certainly be useful. However, for simplicity, the tutorial will stop here.

# Putting it all together
# Now we have code to clean one review - but we need to clean 25,000 training reviews! To make our code reusable, let's create a function that can be called many times:
#vamos hacer una funcion donde se tenga todos estos metodos juntos y no solo pueda procesar un 
#documento si no por ejemplo 25000 documentos y limpiarlos 





