from bs4 import BeautifulSoup  # se utiliza para limpiar paracteres HTML el primer preproseso del texto
# import lxml.html
import re  # libreria para manejo de expreciones regulares
import nltk  # que se utilizara para obtener palabras claves stopwods
#nltk.download()
#nltk.download ( 'punkt'  ) #descargar para hacer uso de la creacion de tokens
from nltk.corpus import stopwords  # Import the stop word list importa las stop words
from nltk.stem.porter import PorterStemmer      #para usar algoritmo Porter Stemming Algorithm
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import RegexpStemmer

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize         #clase para tokenizartexto
from nltk.tokenize import TreebankWordTokenizer #otra clase para tokenizar texto
#from nltk.tokenize import PunktWordTokenizer    #clase para tokenizar contracciones que se encuentren en el texto
from nltk.tokenize import WordPunctTokenizer    #clase para tokenizar puntacion en el texto
from nltk.tokenize import RegexpTokenizer       #para tokenizacion con expreciones regulares



class preposesamiento():
    def __init__(self):
        pass

    def metodoLimpiezaDeMultitextos(self):
        print "Metodo de limpieza de multitextos..."

    def limpiezaDeTextoMuestra(self,textoMuestra):
        print "Limpiando texto seleccionado..."

    def metodoParaLimpiarNnumeroDetextosConMuestreo(self, totaldeTextos, train):

        # Initialize an empty list to hold the clean reviews, crear el vector que guardara todos los archivos
        #lista donde se guardan los textos limpios
        clean_train_reviews = []

        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list 

        for i in xrange(0, totaldeTextos):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            # print i
            # print train["review"][0]
            if ((i + 1) % 5000 == 0):
                print "Review %d of %d\n" % (i + 1, totaldeTextos)
                print train["review"][i]

            clean_train_reviews.append(self.review_to_words2(train["review"][i]))
            # clean_train_reviews.append(preposesamiento.review_to_words2(train["review"][i]))


        return clean_train_reviews

    def metodoParaLimpiarNnumeroDetextosConMuestreoEnsayos(self, totaldeTextos, train, nombreColumna):

        # Initialize an empty list to hold the clean reviews, crear el vector que guardara todos los archivos
        # lista donde se guardan los textos limpios
        clean_train_reviews = []


        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list

        for i in xrange ( 0, totaldeTextos ):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            # print i
            # print train["review"][0]
            if ((i + 1) % 5000 == 0):
                print "Review %d of %d\n" % (i + 1, totaldeTextos)
                print train[nombreColumna][i]

            clean_train_reviews.append ( self.review_to_words2 ( train[nombreColumna][i] ) )
            # clean_train_reviews.append(preposesamiento.review_to_words2(train["review"][i]))

        return clean_train_reviews

    def metodoParaLimpiarNnumeroDetextosConMuestreoEnsayosYStreming2(self, totaldeTextos, train,nombreColumna):

        #lista para guardar los textos limpios
        clean_train_reviews = []
        clean_train_reviews_Stemming = []
        clean_train_reviews_Stemming_lemmaning = []

        #porter_stemmer = PorterStemmer() #instacia de la clase de stemming


        for i in xrange(0, totaldeTextos):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            # print i
            # print train["review"][0]
            if ((i + 1) % 5000 == 0):
                print "Review %d of %d\n" % (i + 1, totaldeTextos)
                #para imp
                print train[nombreColumna][i]
            #review_to_words2 sirvepara limpiar el texto
            clean_train_reviews.append(self.review_to_words2(train[nombreColumna][i]))
            # clean_train_reviews.append(preposesamiento.review_to_words2(train["review"][i]))


        #sentencias para tokenizar los textos de sentecias a palabras
        #for texto in clean_train_reviews:
        #    palabras = word_tokenize(texto)
        #    print palabras



        #para hacer el proceso de Stemming a todos los textos
        tokenizer = TreebankWordTokenizer()  # objeto para tokenizar de otra forma
        # aqui hay 3 stimmnizadores el potter stemming da error con un gran conjunto de textos los otros funcionana bien con podo o muchos textos aun checar el error de potter stimming
        stermer  = PorterStemmer()
        #stermer2 = LancasterStemmer()
        #stermer3 = RegexpStemmer('ing')


        for texto in clean_train_reviews:
            # otra forma de tokenizar
            palabras = tokenizer.tokenize(texto)
            cadena = ""
            for palabra in palabras:
                pal = stermer.stem(palabra)
                #print pal
                cadena = cadena +" "+ pal
                #print  cadena
            clean_train_reviews_Stemming.append(cadena)

        #for i in range (0 , clean_train_reviews_Stemming.__len__()):
        #for textoStem in clean_train_reviews_Stemming:
            #print clean_train_reviews_Stemming[i]
        #    print textoStem


        #para el proceso de lemmanizacion
        lemmatizer = WordNetLemmatizer()
        for texto in clean_train_reviews_Stemming:
            # otra forma de tokenizar
            palabras = tokenizer.tokenize(texto)
            cadena = ""
            for palabra in palabras:
                pal = lemmatizer.lemmatize(palabra)
                # print pal
                cadena = cadena + " " + pal
                # print  cadena
            clean_train_reviews_Stemming_lemmaning.append(cadena)

        for textoStem in clean_train_reviews_Stemming_lemmaning:
            # print clean_train_reviews_Stemming[i]
            print textoStem


        #este objeto sirve para tokenizar contraciones de palabras como Can't  por ejemplo
        #tokenizer = PunktWordTokenizer()
        #palabras = tokenizer.tokenize("Can't is a contraction.")
        #print palabras

        #ejemplo para tokenizar texto y puntaciones de texto
        #tokenizer = WordPunctTokenizer()
        #palabras = tokenizer.tokenize("Can't is a contraction.")
        #print palabras

        #tokenizer = RegexpTokenizer("[\w']+")
        #palabras = tokenizer.tokenize("Can't is a contraction.")
        #print palabras

        #palabras = regexp_tokenize("Can't is a contraction.", "[\w']+")

        #tokenizer = RegexpTokenizer('\s+', gaps=True)
        #tokenizer.tokenize("Can't is a contraction.")
        # algortmos para Stemming Lancaster Stemming Algorithm,  Porter Stemming Algorithm
        #return  clean_train_reviews_Stemming
        return clean_train_reviews_Stemming_lemmaning



    def metodoParaLimpiarNnumeroDetextosConMuestreoEnsayosYStreming(self, totaldeTextos, train,nombreColumna):

        # Initialize an empty list to hold the clean reviews, crear el vector que guardara todos los archivos
        #lista donde se guardan los textos limpios
        clean_train_reviews = []
        clean_train_reviews_with_Stemming = []
        clean_train_reviews_with_Stemming_and_Lemmatizer = []
        porter_stemmer = PorterStemmer ( ) # objeto para crear los stremming
        wnl = WordNetLemmatizer ( ) #objeto para crear le Lemmatizer
        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list

        #tokenizer = RegexpTokenizer("[\w']+") #para tokenizar con expreciones regulares
        #tokenizer = RegexpTokenizer('\s+', gaps=True)

        for i in xrange(0, totaldeTextos):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            # print i
            # print train["review"][0]
            if ((i + 1) % 5000 == 0):
                print "Review %d of %d\n" % (i + 1, totaldeTextos)
                #para imp
                print train[nombreColumna][i]

            clean_train_reviews.append(self.review_to_words2(train[nombreColumna][i]))
            # clean_train_reviews.append(preposesamiento.review_to_words2(train["review"][i]))

        for i in range(0,5):
            print ("1-")
            print (clean_train_reviews[i])
        """
        for i in range(0,len(clean_train_reviews)):
            clean_train_reviews_with_Stemming.append(porter_stemmer.stem(clean_train_reviews[i]))

        for i in range ( 0, 5 ):
            print ("2-")
            print (clean_train_reviews_with_Stemming[i])
        """

        """ metodo 1 para generar stremming a todos las palabras
        for i in range ( 0, len ( clean_train_reviews ) ):
            tokens = nltk.word_tokenize(clean_train_reviews[i])
            str = ""
            for j in range (0, len(tokens)):
                #str = str+" "+porter_stemmer.stem (tokens[j])
                str = str + " " + porter_stemmer.stem ( tokens[j] )
                #str = " ".join(porter_stemmer.stem ( tokens[j] ))

            clean_train_reviews_with_Stemming.append(str)

        print ("-------")
        for i in range(0,5):
            print ("2-")
            print (clean_train_reviews_with_Stemming[i])

        """
        #metodo 2 para hacer stremming a todas las palabras 'punkt' con nltk.download()
        for review in  clean_train_reviews :
            #obtenemos el vector de tokens
            tokens = nltk.word_tokenize ( review )
            #tokens = tokenizer.tokenize(review)
            #tokens = tokenizer.tokenize(review)
            #aplicacmos el stremming a todas las palabras
            singles = [porter_stemmer.stem(plural) for plural in tokens]
            """
            print tokens.__len__()
            stri = ""
            for token in tokens:
                print token
                # str = str+" "+porter_stemmer.stem (tokens[j])
                stri = stri + " " + (str)(porter_stemmer.stem(token))
                # str = " ".join(porter_stemmer.stem ( tokens[j] ))

            print "---"
            tokens = []
            print tokens.__len__()

            clean_train_reviews_with_Stemming.append(stri)
            """
            #ahora solo imprrimimos los tokens
            clean_train_reviews_with_Stemming.append(' '.join(singles))

        print ("-------")
        for i in range ( 0, 5 ):
            print ("2-")
            print (clean_train_reviews_with_Stemming[i])

        #aplicacion lLemmatize se debe descar  se debe descar el corpues 'wordnet' con nltk.download()
        for i in range ( 0, len ( clean_train_reviews_with_Stemming ) ):
            # obtenemos el vector de tokens
            tokens = nltk.word_tokenize (clean_train_reviews_with_Stemming[i] )
            # aplicacmos el stremming a todas las palabras
            singles2 = [wnl.lemmatize( plural ) for plural in tokens]
            # ahora solo imprrimimos los tokens
            clean_train_reviews_with_Stemming_and_Lemmatizer.append ( ' '.join ( singles2 ) )

        print ("-------")
        for i in range ( 0, 5 ):
            print ("3-")
            print (clean_train_reviews_with_Stemming_and_Lemmatizer[i])

        #return clean_train_reviews_with_Stemming  #retornar el vector ya convetido el vector con palabras streaming
        return clean_train_reviews_with_Stemming_and_Lemmatizer #retornar el vector convertido el vector con palabras con lemmatizar


    def metodoParaLimpiarNnumeroDetextos(self, totaldeTextos, train):

        # Initialize an empty list to hold the clean reviews, crear el vector que guardara todos los archivos 
        clean_train_reviews = []

        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list 

        for i in xrange(0, totaldeTextos):
            # print(i)
            # Call our function for each one, and add the result to the list of
            # clean reviews
            # print i
            # print train["review"][i]
            # se hace uso de la sentencia self para mandar a traer el metodo de una misma clase
            # You need to call self.a() to invoke a from b. a is not a global function, it is a method on the class.
            # You may want to read through the Python tutorial on classes some more to get the finer details down.
            # if i == 0:


            clean_train_reviews.append(self.review_to_words2(train["review"][i]))
            # clean_train_reviews.append(preposesamiento.review_to_words2(train["review"][i]))


            # if i == totaldeTextos:
            #     clean_train_reviews.append(self.review_to_words2( train["review"][i] ) )
            #     #clean_train_reviews.append(preposesamiento.review_to_words2(train["review"][i]))

    # def __init__(self):
    #     print ("hola nundo ")

    # la siguiente funcion recive como parametros un conjunto de texto
    # el cual sera limpiado primeramente se quitan etiquetas HTML
    # despues se limpian caracteres que no sean lean letras de a A z en minuscula  y matuscula y se colocan espacios en blanco
    # se convierten a letras minusculas cualquier caracter
    # se elimianan stop words
    # y se retorna el texto limpio
    # Two elements here are new: First, we converted the stop word list to a different data type, a set. This is for speed; since we'll be calling this function tens of thousands of times, it needs to be fast, and searching sets in Python is much faster than searching lists.
    # Second, we joined the words back into one paragraph. This is to make the output easier to use in our Bag of Words, below. After defining the above function, if you call the function for a single review:
    # esta funcion sirva para un solo texto , si se desean mas meterla en un for o while
    def review_to_words2(self, raw_review):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        # review_text = BeautifulSoup(raw_review).get_text()  # vercion original

        # otras verciones
        # review_text = BeautifulSoup(raw_review).get_text()
        # review_text = BeautifulSoup(raw_review.get_text(), 'html.parser')
        # review_text = BeautifulSoup( raw_review.get_text() )
        # tmpTexto = raw_review.get_text()


        # textoLimpo = textoHTML.get_text()
        # print(textoLimpo.encode("utf-8"))

        # textoLimpo = BeautifulSoup(raw_review).get_text()
        # BeatifulSoup( ..., "html.parser")
        #textoAlimpiar = raw_review.get_text()
        #limpio = BeautifulSoup(textoAlimpiar, 'html.parser')

        texto = raw_review
        pal = BeautifulSoup (texto, "lxml" ).get_text ()            #funcion para sacar el texto si es que se tiene etiqutas html
        #review_text = BeautifulSoup(raw_review).get_text()

        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", pal)                #se iltran solo latras mayusculas y minusculas

        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()                        #se pasan las palabras a minusculas
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))  #mandar a traer las palabras stopwords de donde tomar referencia, se debe tener el corpues de stopwords de  libreria from nltk.corpus import stopwords
        #stops.fileids() para ver la lista de stopwords
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stops]     # esta sentencia es para filtar y quitar  las palabras stopwords
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        return (" ".join(meaningful_words))                         # se genera el texto limpio listo para retornar

    def metodoejemplo1(self):
        print ("hola mundo")


    # la siguiente funcion recive como parametros un conjunto de texto
    # el cual sera limpiado primeramente se quitan etiquetas HTML
    # despues se limpian caracteres que no sean lean letras de a A z en minuscula  y matuscula y se colocan espacios en blanco
    # se convierten a letras minusculas cualquier caracter
    # se elimianan stop words
    # y se retorna el texto limpio
    # Two elements here are new: First, we converted the stop word list to a different data type, a set. This is for speed; since we'll be calling this function tens of thousands of times, it needs to be fast, and searching sets in Python is much faster than searching lists.
    # Second, we joined the words back into one paragraph. This is to make the output easier to use in our Bag of Words, below. After defining the above function, if you call the function for a single review:

    # NOTA recordar que siempre el primer parametro para cual quier metodo debe tener un paramentro self
    def review_to_words(self, raw_review):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(raw_review.get_text(), 'html.parser')
        # 2. Remove non-letters
        # otra exprecion regular "ur'(?![\d_])\w'"
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stops]
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        return (" ".join(meaningful_words))

    # estructura de ejemplo de metodo sin ningun pase de paramentros siempre debe llevar el parametro self
    def metodoejemplo1(self):
        print ("hola mundo")


    # estructura de ejemplo de metodo sin ningun pase de paramentros siempre debe llevar el parametro self
    def metodoejemplo2(self, num):
        print ("hola mundo", num)


    # estructura de ejemplo de metodo sin ningun pase de paramentros siempre debe llevar el parametro self
    def metodoejemplo3(self, num):
        print ("hola mundo", num)
        return num + num
