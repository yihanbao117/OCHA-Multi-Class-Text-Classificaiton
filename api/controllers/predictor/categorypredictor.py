#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

import pandas as pd
import json
import sys
import os
import ast
from ast import literal_eval

from flask_restful import Resource  # Flask restful for create endpoints
from flask import request, jsonify
from run import app 
from helper import Helper as ett_h  # ETT helper package
from transformer import Transformer as ett_t  # ett transfer package
ocha_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),"classification")
sys.path.append(ocha_path)
from ocha import TextClassification as tc
# Initiate Parameter
base_folder_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),"category")

colnames = ['title', 'text']
label_dict = {6863: "Human Resources", 6864:"Administration/Finance", 6865:"Advocacy/Communications", 
          6866:"Information and Communications Technology", 6867: "Program/Project Management", 
          6868:"Monitoring and Evaluation", 20966:"Donor Relations/Grants Management", 
          20971:"Information Management", 36601:"Logistics/Procurement"}
# Folder name
vector_models_folder = "vector_models"
normalizar_models_folder = "normalizar_models"
models_folder = "models"

# Model File name 
vector_model_name = 'vectorizar.pickle'
normalizar_model_name = 'normalizar.pickle'
model_name = 'model.pickle'

# New Text Models 
models_object = []

@app.before_first_request
def load_models():
    # load all the models: vectorizar, normalizar and models for career category
    vector_filepath = ett_h.generate_dynamic_path([base_folder_location, vector_models_folder,vector_model_name])
    normalizar_filepath = ett_h.generate_dynamic_path([base_folder_location, normalizar_models_folder,normalizar_model_name])
    models_filepath = ett_h.generate_dynamic_path([base_folder_location, models_folder,model_name])

    vector_model = ett_h.load_model(vector_filepath)
    print("vector_model",vector_model)
    normalizar_model = ett_h.load_model(normalizar_filepath)
    print("normalizar_model",normalizar_model)
    model = ett_h.load_model(models_filepath)
    print("model",model)
    global models_object
    models_object = {"vector_model":vector_model,"normalizar_model":normalizar_model,"prediction_model":model}
    #models_object = [vector_model, normalizar_model, model]

class UploadCategory(Resource):
   
    
    # Global the data_df parameters
    def __init__(self):
    
        UploadCategory.data_df = " "
        data_df = ''

    def post(self):

        bytes_data = request.stream.read()
        str_data = bytes_data.decode("utf-8")
        dic_data = ast.literal_eval(str_data)
        global input_data
        input_data = pd.DataFrame.from_dict(dic_data) # change dictionary to dataframe
        UploadCategory.data_df = ett_t.transform_data_to_dataframe_basic(input_data, colnames)
    
        #return results
        return jsonify({'message' : dic_data})
        #return "uploading successfully"

class PredictCategory(Resource):


    def get(self):
        
        classification = tc(models_object,UploadCategory.data_df)
        result_df = classification.process_data()
        print("yihan",type(result_df))
        result_df.reset_index(drop=True, inplace=True)
        input_data.reset_index(drop=True, inplace=True)
        result_final = pd.concat([input_data, result_df], axis=1)
        result_final['label'] = result_final['code'].map(label_dict)     
        print(result_final)
        result_str = ett_t.df_to_json(result_final,'index')
        result_json = literal_eval(result_str)
        return result_json
  
        