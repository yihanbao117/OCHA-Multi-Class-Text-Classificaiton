#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

import re  # Package used to as replace function
import numpy as np  # Mathematic caculations
import pandas as pd  # Dataframe operations
from flask import jsonify # flask package
from cleanser import Cleanser as ett_c  # ETT Cleanser methods
from predictor import Predictor as ett_p  # ETT Predictor methods
from transformer import Transformer as ett_t  # ETT Transformer methods
from constants import RegexFilter  # ETT constants methods for RegexFilter
from constants import Language  # ETT constants methods for language type
from enum import Enum  # Custom enum for classification type

class TextClassification:


    models = []
    data = pd.DataFrame()
    #output_dataFrame = pd.DataFrame()  # dataframe

    def __init__(self, models, data):
        self.models = models
        self.data = data

    # Entry point method to actually start the
    # classification operations
    def process(self):
        self.process_data()

    # Method which acts as the builder
    def process_data(self):
        
        # Clean the data
        cleansed_data = self.pre_process_text_cleanse(self.data)

        # Transform the data
        transformed_data = self.pre_process_text_transform(cleansed_data)
    
        # vectorize the data
        vectorized_data = ett_t.perform_model_transformation(self.models["vector_model"], transformed_data)

        # Normalize teh data
        normalized_data = ett_t.perform_model_transformation(self.models["normalizar_model"], vectorized_data)
        
        # Get the predciton labels
        labelled_data = ett_p.perform_model_predictions(self.models["prediction_model"], normalized_data)
        result_label = pd.DataFrame(labelled_data)
   
        # Get the probabilities dataframe
        probabilities_data = ett_p.perform_model_prob_predictions(self.models["prediction_model"], normalized_data)
        result_prob = pd.DataFrame(probabilities_data)
        
        # get the probability of prediction
        result_prob['max_value'] = result_prob.max(axis=1)
        # A list of dataframe that containes all columns you want to show in UI
        result_list = [result_label,result_prob['max_value']]
        results = ett_t.combine_dataframe(result_list,1)
        results.columns = ['label', 'proba']
        return results

    def pre_process_text_cleanse(self, initial_data):
        # Removed all non alphanumeric characters
        nan_cleaned_data = ett_c.clean_dataframe_by_regex(initial_data, RegexFilter.NON_ALPHA_NUMERIC.value) 
        # Removed all digits
        d_cleaned_data = ett_c.clean_dataframe_by_regex(nan_cleaned_data, RegexFilter.DIGITS_ONLY.value) 
        # Remove non-English text
        l_cleaned_data = ett_c.remove_non_iso_words(d_cleaned_data, Language.ENGLISH.value)  
        # Remove English stop words
        rew_cleaned_data = ett_c.remove_language_stopwords(l_cleaned_data, Language.ENGLISH.name)
        # Remove HTML element
        rew_cleaned_data =rew_cleaned_data.apply(lambda x: re.sub('<[^>]+>', "",x))
        # Remove Special Character
        rew_cleaned_data = rew_cleaned_data.apply(lambda k: re.sub(r"[^a-zA-Z0-9]+", ' ', k))
        # Remove Number 
        rew_cleaned_data = rew_cleaned_data.apply(lambda c: re.sub(" \d+", " ", c))
        # Return the newly cleaned data
        return rew_cleaned_data 

    ##
    # Method used to contain the various transformation procedures performed on the data
    # @param cleaned_data DataFrame of the cleansed data
    # @returns DataFrame of the transformed data according to these rules
    def pre_process_text_transform(self, cleaned_data):
        # Transform text to lowercase
        l_transformed_data = ett_t.lowercase(cleaned_data)
        # Transform text to core words i.e. playing > play
        le_transformed_data = ett_t.lemmatization_mp(l_transformed_data)
        # Return the newly transformed data
        return le_transformed_data