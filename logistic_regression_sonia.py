#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Sonia Regression
    Copyright (C) February 2022 Francesco Camaglia, LPENS 
    Adaptations from Giulio Isacchini : https://github.com/statbiophys/soNNia/blob/6d99a55cb8c6b71f0ef110f1eefccbd71f789d8d/sonnia/classifiers.py
'''

import os 
import numpy as np
import pandas as pd

# to avoid warning due to encoding sonia features
import warnings
warnings.filterwarnings("ignore")
# avoid tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
from tensorflow import keras

# >>>>>>>>>>>>>>>>>>>>>>>
#  BINARY DATA PROCESS  #
# >>>>>>>>>>>>>>>>>>>>>>>

class Binary_Data :
    '''
    A class to merge two pandas dataframe and divide them in training and testing.
    '''
    
    def __init__( self, df_Pos=None, df_Neg=None, PERC=0.75, 
        return_sharing=False, max_size_ratio=1.5, upper_bound=None, load=None ) :
        
        if load :
            self.load( load )
        else :

            # Look for Shared sequences
            merged = df_Pos.merge( df_Neg, how="outer", indicator=True )
            both = merged.copy().loc[ lambda x : x['_merge']=='both' ]
            both.drop( labels="_merge", axis = 1, inplace = True )
            self.shared = both.values

            # delete shared sequences between the datasets
            if return_sharing is False :

                # redefine data excluding shared sequences
                df_Pos = merged.copy().loc[ lambda x : x['_merge']=='left_only' ]
                df_Pos.drop(labels="_merge", axis = 1, inplace = True)
                #
                df_Neg = merged.copy().loc[ lambda x : x['_merge']=='right_only' ]
                df_Neg.drop(labels="_merge", axis = 1, inplace = True)

            #
            # choice of the training data size    
            #

            Sizes = [len(df_Pos), len(df_Neg) ]
            min_size = np.min(Sizes)

            if upper_bound == None :  

                # downsample according to the chosen parameter
                frac_datasets = np.max(Sizes) / min_size
                # check that the chosen parameter is valid
                if max_size_ratio > frac_datasets : max_size_ratio = frac_datasets
                # get vector with train sizes 
                final_sizes = min_size * np.ones( 2 )
                final_sizes *= ( Sizes != min_size ) * max_size_ratio + ( Sizes == min_size )
                final_sizes = np.floor(final_sizes).astype(int)    

            elif upper_bound > 0 :
                final_sizes = (np.min( [ min_size, upper_bound ] ) * np.ones(2)).astype(int)
                final_sizes = final_sizes.astype(int)

            train_sizes = (PERC * final_sizes).astype(int)

            # Positive data
            rand_indx = np.arange( Sizes[0] )
            np.random.shuffle( rand_indx )
            self.positive = np.split( df_Pos.iloc[ rand_indx ].values, [ train_sizes[0], final_sizes[0] ] )[:2]

            # Negative data 
            rand_indx = np.arange( Sizes[1] )
            np.random.shuffle( rand_indx )
            self.negative = np.split( df_Neg.iloc[ rand_indx ].values, [ train_sizes[1], final_sizes[1] ] )[:2]   
    ###        
    
    # >>>>>>>>>>
    #  saving  #
    # >>>>>>>>>>
    
    def save( self, outpath ):
        '''
        Where to save the dataframe
        '''
        df = pd.DataFrame()

        attach = pd.DataFrame(self.positive[0])
        attach[['Label','Use']] = '1', 'train'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.positive[1])
        attach[['Label','Use']] = '1', 'test'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.negative[0])
        attach[['Label','Use']] = '0', 'train'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.negative[1])
        attach[['Label','Use']] = '0', 'test'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.shared)
        attach[['Label','Use']] = 'shared', 'shared'
        df = df.append(attach, ignore_index=True)
        
        # saving
        df.to_csv( f'{outpath}/binary_data.csv.gz', sep=',', index=False, compression='gzip' )
        
    # >>>>>>>>>>
    #  loading  #
    # >>>>>>>>>>
    
    def load( self, outpath ):
        '''
        Where to load the dataframe
        '''
        df = pd.read_csv( f'{outpath}/binary_data.csv.gz', compression='gzip', low_memory=False, dtype=str )
        Pos_msk = df['Label'] == '1'
        Neg_msk = df['Label'] == '0'
        Shared_msk = df['Label'] =='shared'
        Train_msk = df['Use'] == 'train'
        Test_msk = df['Use'] == 'test'
        df.drop(columns=['Use', 'Label'], inplace=True)
        positive_train = df[np.logical_and(Pos_msk, Train_msk)].values
        positive_test = df[np.logical_and(Pos_msk, Test_msk)].values
        self.positive = [positive_train, positive_test]
        
        negative_train = df[np.logical_and(Neg_msk, Train_msk)].values
        negative_test = df[np.logical_and(Neg_msk, Test_msk)].values  
        self.negative = [negative_train, negative_test]
            
        self.shared = df[Shared_msk].values

        del df

    # >>>>>>>>>>>>>>>>>>>
    #  train/test sets  #
    # >>>>>>>>>>>>>>>>>>>

    def train( self ) :
        x_train = np.concatenate( [ self.positive[0], self.negative[0] ] )
        y_train = np.append( np.ones(len(self.positive[0])), np.zeros(len(self.negative[0])) )
        return x_train, y_train

    def test( self ) :
        x_test = np.concatenate( [ self.positive[1], self.negative[1] ] )
        y_test = np.append( np.ones(len(self.positive[1])), np.zeros(len(self.negative[1])) )
        return x_test, y_test
###



#############################
#  CLASSIFY ON SONIA CLASS  #
#############################

class ClassifyOnSonia( object ):
    '''
    Logistic Regression over sonia features
    '''

    # >>>>>>>>>>>>>>
    #  INITIALIZE  #
    # >>>>>>>>>>>>>>

    def __init__( self, which_sonia_model="leftright", load_model=None, custom_pgen_model=None, 
                 vj=False, include_indep_genes=True, include_joint_genes=False ) :
        
        
        if load_model is not None :
            # load model from directory
            self.sonia_model = SoniaLeftposRightpos( load_dir = load_model )
            n_features = len(self.sonia_model.features)
            
        elif which_sonia_model == "leftright" :
            # Default Sonia Left to Right Position model
            self.sonia_model = SoniaLeftposRightpos( custom_pgen_model=custom_pgen_model, vj=vj,
                                                    include_indep_genes=include_indep_genes,
                                                    include_joint_genes=include_joint_genes ) 
            n_features = len(self.sonia_model.features)

        else :            
            raise IOError('Unknwon option for `which_sonia_model`.')
        
        self.which_sonia_model = which_sonia_model
        
        # Number of feautures associated to each sequence according to the model
        self.input_size = n_features
        
    ###
    
    '''
    Methods
    '''
               
    # >>>>>>>>>>
    #  encode  #
    # >>>>>>>>>>

    def encode( self, aa_V_J ):
        '''
        Extract features from sequence in `aa_V_J` according to sonia model
        '''
        
        aa_V_J = np.array(aa_V_J)
        
        if self.which_sonia_model in ["alpha+beta"] :
            data = list(map(lambda x : self.sonia_model[0].find_seq_features(x[0:3]) + self.sonia_model[1].find_seq_features(x[4:-1]) , aa_V_J))
        else :     
            data = list(map(lambda x : self.sonia_model.find_seq_features(x), aa_V_J))
            
        data = np.array( data )
        data_enc = np.zeros( ( len(data), self.input_size ), dtype=np.int8 )
        for i in range( len(data_enc) ): 
            data_enc[ i ][ data[ i ] ] = 1
        '''
        # delete non informative features
        # delete all emty feature columns 
        data_enc = data_enc[:, ~np.all(data_enc == 0, axis = 0)]
        # delete all full feature columns
        data_enc = data_enc[:, ~np.all(data_enc == 1, axis = 0)]
        '''
        return data_enc
###



###############################
#  LINEAR LEFT POS RIGHT POS  #
###############################

# mono layer logistic regression on
class Logistic_Sonia_LeftRight( ClassifyOnSonia ):
    '''
    Logistic Regression over sonia features (i.e. activation='sigmoid')
    '''
                
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
    #  update_model_structure  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>

    def update_model_structure( self, input_layer=None, output_layer=None, 
                               initialize=True, activation='sigmoid', 
                               optimizer='Adam', loss='binary_crossentropy', 
                               metrics=["binary_accuracy"] ) :
        
        if initialize is True:
            # Initiliaze ML model layers which bring from n. features to 2 possibilities (categorical)
            input_layer = keras.layers.Input( shape = (self.input_size,) )
            output_layer = keras.layers.Dense( 1, activation=activation )( input_layer )

        # Define model from the specified layers 
        self.model = keras.models.Model( inputs=input_layer, outputs=output_layer )

        # Once the model is created it is then configurated with losses and metrics 
        self.model.compile( optimizer=optimizer, loss=loss, metrics=metrics )
    
    def fit( self, x, y, batch_size=300, epochs=100, val_split=0 ) :
        '''
        Fit the keras supervised model on data x with label y encoding features of x to x_enc.
        It shuffles data automatically.
        '''
        
        x_enc = self.encode( x )
        
        # shuffle indeces
        rand_indx = np.arange( len(y) )
        np.random.shuffle( rand_indx )
        # fit to the model
        self.history = self.model.fit( x_enc[ rand_indx ], y[ rand_indx ],
                                      batch_size=batch_size, epochs=epochs,
                                      verbose=0, validation_split=val_split )

    def predict( self, x ):
        x_enc = self.encode( x )
        return self.model.predict( x_enc )
    
    def save( self, outpath ) :
        self.model.save( outpath )
    
    def load_model( self, outpath ) :
        self.model = keras.models.load_model( outpath )
        
###