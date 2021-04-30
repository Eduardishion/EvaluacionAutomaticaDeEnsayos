
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#para seleccion de caracteristicas
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#para seleccion de caracteristicas
from sklearn.ensemble import RandomForestClassifier
from seleccionCaracteristicas import boruta_py


import numpy as np

# -*- coding: utf8 -*-
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.



class bolsaDePalabras():

    def obtenerFrecuenciaDeTerminostf(self, train_data_features_res_tmp):

        tf_transformer = TfidfTransformer ( use_idf=False ).fit ( train_data_features_res_tmp )
        frecuenciaDeTerminos = tf_transformer.transform ( train_data_features_res_tmp )

        return frecuenciaDeTerminos


    def obtenerFrecuenciaInversaDelDocumentos(self, train_data_features_res_tmp):
        tfidf_transformer = TfidfTransformer ( )
        frecuenciaDeTerminos = tfidf_transformer.fit_transform (train_data_features_res_tmp)

        return frecuenciaDeTerminos

    def obtenerBolsaDePalabrasMejoresCaracteristicas(self, clean_train_reviewsTmp, targets):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer (
            analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=150)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array

        train_data_features_mejores = SelectKBest ( chi2, k=50 ).fit_transform ( train_data_features, targets )

        print ("caracteristicas: ", train_data_features_mejores.shape)


        train_data_features_res = train_data_features_mejores.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # for tag, count in zip(vocab, dist):
        # print count, tag

        # lo mismo que lo anterior pero con funcion
        # self.mostrarLascaracteristicas(vectorizer,train_data_features)

        return train_data_features_res

    def obtenerCaracteristicas(self, clean_train_reviewsTmp):
        vec = DictVectorizer ()
        train_data_features = vec.fit_transform ( clean_train_reviewsTmp )
        print ("caracteristicas: ", train_data_features.shape)
        train_data_features_res = train_data_features.toarray ( )
        return train_data_features_res



    def obtenerBolsaDePalabras(self, clean_train_reviewsTmp):

        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer(analyzer='word', tokenizer=None,  preprocessor=None, stop_words=None ,max_features=60)


        #max_features=60,ngram_range=(1, 3)

        #vectorizer = HashingVectorizer( max_features=50)
        #

        # CountVectorizer ( analyzer='word', binary = False, decode_error = 'strict', dtype = < 'numpy.int64' >, encoding = 'utf-8', input = 'content', lowercase = True, max_df = 1.0, max_features = None, min_df = 1,ngram_range = (1,1), preprocessor = None, stop_words = None,strip_accents = None, token_pattern = '(?u)\\b\\w\\w+\\b',tokenizer = None, vocabulary = None)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform(clean_train_reviewsTmp)



        print ("caracteristicas: ",train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array


        train_data_features_res = train_data_features.toarray()

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
        # appears in the training set
    # print count, tag
        # for tag, count in zip(vocab, dist):

    # lo mismo que lo anterior pero con funcion
        self.mostrarLascaracteristicas(vectorizer,train_data_features)

        return train_data_features_res

    #def seleccionDeCaracteristicas(self,train_data_features, ):



    def extraccionDeCaracteristicasBoW(self, clean_train_reviewsTmp):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                       max_features=60 )

        # max_features=60,ngram_range=(1, 3)

        # vectorizer = HashingVectorizer( max_features=50)
        #

        # CountVectorizer ( analyzer='word', binary = False, decode_error = 'strict', dtype = < 'numpy.int64' >, encoding = 'utf-8', input = 'content', lowercase = True, max_df = 1.0, max_features = None, min_df = 1,ngram_range = (1,1), preprocessor = None, stop_words = None,strip_accents = None, token_pattern = '(?u)\\b\\w\\w+\\b',tokenizer = None, vocabulary = None)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array


        train_data_features_res = train_data_features.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # print count, tag
        # for tag, count in zip(vocab, dist):

        # lo mismo que lo anterior pero con funcion
        self.mostrarLascaracteristicas ( vectorizer, train_data_features )
        return train_data_features_res

    def extraccionDeCaracteristicasBoW2(self, clean_train_reviewsTmp):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        #elegir este si seva ocupar el selector de caracteristicas
        #vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
        #                                                                                   max_features=250)#60 caracteristicas son las optimas sin mandar a traer el selecctor de caracteristicas
        vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None
                                                                                           ,max_features=300)# 60 caracteristicas son las optimas sin mandar a traer el selecctor de caracteristicas

        # max_features=60,ngram_range=(1, 3)

        # vectorizer = HashingVectorizer( max_features=50)
        #

        # CountVectorizer ( analyzer='word', binary = False, decode_error = 'strict', dtype = < 'numpy.int64' >, encoding = 'utf-8', input = 'content', lowercase = True, max_df = 1.0, max_features = None, min_df = 1,ngram_range = (1,1), preprocessor = None, stop_words = None,strip_accents = None, token_pattern = '(?u)\\b\\w\\w+\\b',tokenizer = None, vocabulary = None)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array


        train_data_features_res = train_data_features.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # print count, tag
        # for tag, count in zip(vocab, dist):

        # lo mismo que lo anterior pero con funcion
        self.mostrarLascaracteristicas ( vectorizer, train_data_features )
        return train_data_features_res


    def extraccionDeCaracteristicasTfidf(self, clean_train_reviewsTmp):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        #elegir este si seva ocupar el selector de caracteristicas
        #vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
        #                                                                                   max_features=250)#60 caracteristicas son las optimas sin mandar a traer el selecctor de caracteristicas
        vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                                                                           max_features=60)#60 caracteristicas son las optimas sin mandar a traer el selecctor de caracteristicas





        # max_features=60,ngram_range=(1, 3)

        # vectorizer = HashingVectorizer( max_features=50)
        #

        # CountVectorizer ( analyzer='word', binary = False, decode_error = 'strict', dtype = < 'numpy.int64' >, encoding = 'utf-8', input = 'content', lowercase = True, max_df = 1.0, max_features = None, min_df = 1,ngram_range = (1,1), preprocessor = None, stop_words = None,strip_accents = None, token_pattern = '(?u)\\b\\w\\w+\\b',tokenizer = None, vocabulary = None)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array


        train_data_features_res = train_data_features.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # print count, tag
        # for tag, count in zip(vocab, dist):

        # lo mismo que lo anterior pero con funcion
        self.mostrarLascaracteristicas ( vectorizer, train_data_features )
        return train_data_features_res


    def seleccionDeCaracteristicasSelectKBest(self, train_data_features_res,train,nombreColumna):

        new_train_data_features_res = SelectKBest ( chi2, k=60 ).fit_transform ( train_data_features_res , train[nombreColumna])
        return new_train_data_features_res



    def seleccionDeCaracteristicas(self,train_data_features_res,train,nombreColumna):
        # define random forest classifier, with utilising all cores and
        # sampling in proportion to y labels
        rf = RandomForestClassifier ( n_jobs=-1, class_weight='auto', max_depth=5)

        # define Boruta feature selection method
        feat_selector = BorutaPy ( rf, n_estimators=50, verbose=2, random_state=1 )

        # find all relevant features - 5 features should be selected
        feat_selector.fit ( train_data_features_res , train[nombreColumna])

        # check selected features - first 5 features are selected
        feat_selector.support_

        # check ranking of features
        feat_selector.ranking_

        # call transform() on X to filter it down to selected features
        train_data_features_res_selectas = feat_selector.transform (train_data_features_res)

        return train_data_features_res_selectas

    def seleccionDeCaracteristicasRemovingFeaturesLowVariance(self, train_data_features_res):
        sel = VarianceThreshold ( threshold=(.8 * (1 - .8)) )
        new_train_data_features_res = sel.fit_transform (train_data_features_res)

        return new_train_data_features_res





    def obtenerBolsaDePalabrasNgramas(self, clean_train_reviewsTmp):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                       max_features=60,ngram_range=(1, 3),token_pattern = r'\b\w+\b', min_df = 1 )

        # max_features=60,ngram_range=(1, 3)

        # vectorizer = HashingVectorizer( max_features=50)
        #

        # CountVectorizer ( analyzer='word', binary = False, decode_error = 'strict', dtype = < 'numpy.int64' >, encoding = 'utf-8', input = 'content', lowercase = True, max_df = 1.0, max_features = None, min_df = 1,ngram_range = (1,1), preprocessor = None, stop_words = None,strip_accents = None, token_pattern = '(?u)\\b\\w\\w+\\b',tokenizer = None, vocabulary = None)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array


        train_data_features_res = train_data_features.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # print count, tag
        # for tag, count in zip(vocab, dist):

        # lo mismo que lo anterior pero con funcion
        self.mostrarLascaracteristicas ( vectorizer, train_data_features )

        return train_data_features_res

    def obtenerBolsaDePalabras_Tf_idf(self, clean_train_reviewsTmp):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                       max_features=120,token_pattern = r'\b\w+\b', min_df = 1 )

        # max_features=60,ngram_range=(1, 3)

        # vectorizer = HashingVectorizer( max_features=50)
        #

        # CountVectorizer ( analyzer='word', binary = False, decode_error = 'strict', dtype = < 'numpy.int64' >, encoding = 'utf-8', input = 'content', lowercase = True, max_df = 1.0, max_features = None, min_df = 1,ngram_range = (1,1), preprocessor = None, stop_words = None,strip_accents = None, token_pattern = '(?u)\\b\\w\\w+\\b',tokenizer = None, vocabulary = None)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array


        train_data_features_res = train_data_features.toarray ( )

        transformer = TfidfTransformer(norm='l2', smooth_idf=False, sublinear_tf=False,use_idf=True)

        tfidf = transformer.fit_transform (train_data_features_res)

        train_data_features_res_Tf_idf = tfidf.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # print count, tag
        # for tag, count in zip(vocab, dist):

        # lo mismo que lo anterior pero con funcion
        self.mostrarLascaracteristicas ( vectorizer, train_data_features )

        return train_data_features_res_Tf_idf

    def obtenerBolsaDePalabras_with_TfidfVectorizer(self, clean_train_reviewsTmp):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.


        vectorizer = CountVectorizer ( analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                       max_features=120, token_pattern=r'\b\w+\b', min_df=1 )


        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )



        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array

        train_data_features_res = train_data_features.toarray ( )



        transformer = TfidfVectorizer ( min_df=1 )

        tfidf = transformer.fit_transform ( train_data_features_res )

        train_data_features_res_with_TfidfVectorizer = tfidf.toarray ( )

        # max_features=60,ngram_range=(1, 3)

        # vectorizer = HashingVectorizer( max_features=50)
        #

        # CountVectorizer ( analyzer='word', binary = False, decode_error = 'strict', dtype = < 'numpy.int64' >, encoding = 'utf-8', input = 'content', lowercase = True, max_df = 1.0, max_features = None, min_df = 1,ngram_range = (1,1), preprocessor = None, stop_words = None,strip_accents = None, token_pattern = '(?u)\\b\\w\\w+\\b',tokenizer = None, vocabulary = None)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.




        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # print count, tag
        # for tag, count in zip(vocab, dist):

        # lo mismo que lo anterior pero con funcion
        self.mostrarLascaracteristicas ( vectorizer, train_data_features )

        return train_data_features_res_with_TfidfVectorizer


    def obtenerBolsaDePalabrasEnt(self, clean_train_reviewsTmp):
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer (
            analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=3000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        print ("caracteristicas: ", train_data_features.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array
        train_data_features_res = train_data_features.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # for tag, count in zip(vocab, dist):
        # print count, tag

        # lo mismo que lo anterior pero con funcion
        # self.mostrarLascaracteristicas(vectorizer,train_data_features)

        return train_data_features_res

    def obtenerBolsaDePalabras2(self, clean_train_reviewsTmp):

        tfidf_transformer = TfidfTransformer ()
        print ("Creating the bag of words...\n")

        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer(analyzer='word', tokenizer=None,  preprocessor=None, stop_words=None,   max_features=1000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform ( clean_train_reviewsTmp )

        train_data_features_tfidf = tfidf_transformer.fit_transform (train_data_features)



        print ("caracteristicas: ",train_data_features.shape)
        print ("caracteristicas2: ",train_data_features_tfidf.shape)
        # Numpy arrays are easy to work with, so convert the result to an
        # array
        #train_data_features = train_data_features.toarray()
        train_data_features = train_data_features_tfidf.toarray ( )

        # train_data_features.shape
        # print train_data_features.shape

        # Take a look at the words in the vocabulary  Now that the Bag of Words model is trained, let's look at the vocabulary:
        ##vocab = vectorizer.get_feature_names()
        # print vocab

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        ##dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
        # appears in the training set
        # for tag, count in zip(vocab, dist):
    # print count, tag

    # lo mismo que lo anterior pero con funcion
        # self.mostrarLascaracteristicas(vectorizer,train_data_features)

        return train_data_features


    def mostrarLascaracteristicas(self, Elemtovectorizer, train_data_featuresTmp):
        # Take a look at the words in the vocabulary  Now that the Bag of Words
        # model is trained, let's look at the vocabulary:
        vocab = Elemtovectorizer.get_feature_names()
        print (vocab)

        # If you're interested, you can also print the counts of each word in the vocabulary:
        # Sum up the counts of each vocabulary word
        dist = np.sum(train_data_featuresTmp, axis=0)
        # For each, print the vocabulary word and the number of times it
        # appears in the training set

        for tag, count in zip(vocab, dist):
            print (count, tag)

        print (train_data_featuresTmp.shape)
