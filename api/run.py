#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

"""     
This script is the entry point of running the OCHA Text Classification web application.
""" 

from flask import Flask
from flask_restful import Api
from flask import jsonify
from flask import request
import os
from flask import Flask, request, jsonify
import pandas as pd
import sys
import pandas as pd
import json
from flask import flash, redirect
from pyxlsb import open_workbook
db_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'ett')
sys.path.append(db_folder)

# The main entry point for the application

app = Flask(__name__)
app.secret_key = "acbdefg"
api = Api(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message' : 'OCHA FLASK API started'})

from controllers.predictor import categorypredictor
#from controllers.predictor import themepredictor

# Add a resource to the api
api.add_resource(categorypredictor.UploadCategory, '/upload/category')
api.add_resource(categorypredictor.PredictCategory, '/predict/category')