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
from textblob import Word  # Package used to do lemmatization
from nltk.corpus import stopwords # Corpus for English stopwords

class TextClassification:


    models = []
    data = pd.DataFrame()

    def __init__(self, models, data):
        self.models = models
        self.data = data

    # Entry point method to actually start the classification operations
    def process(self):
        self.process_data()

    # Method which acts as the builder
    def process_data(self):
        
        # Clean and transform the text data
        cleansed_data = self.pre_process_text_cleanse_transform(self.data)
        # vectorize the data
        vectorized_data = ett_t.perform_model_transformation(self.models["vector_model"], cleansed_data)
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
        results.columns = ['code', 'proba']
        return results

    def pre_process_text_cleanse_transform(self, initial_data):
        # Remove HTML element
        rew_cleaned_data =initial_data.apply(lambda x: re.sub('<[^>]+>', "",x))
        # Remove Special Character
        rew_cleaned_data = rew_cleaned_data.apply(lambda k: re.sub(r"[^a-zA-Z0-9]+", ' ', k))
        # Removed all digits
        rew_cleaned_data = rew_cleaned_data.apply(lambda c: re.sub(" \d+", " ", c))
        # Lowercase the words
        rew_cleaned_data = rew_cleaned_data.apply(lambda x: " ".join(x.lower() for x in x.split()))
        # Text Normalization lemmatization
        rew_cleaned_data = rew_cleaned_data.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        # Remove English stop words
        stops = set(stopwords.words("english"))
        rew_cleaned_data = rew_cleaned_data.apply(lambda x: " ".join(x for x in x.split() if x not in stops))
        # Return the newly cleaned data
        return rew_cleaned_data 