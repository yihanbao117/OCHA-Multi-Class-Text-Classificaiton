#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library housing ENUM lists
    
    Contains functions that focus on Enum
"""

__author__ = "Yihan Bao"
__copyright__ = "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "11 February 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

from enum import Enum  # Used for custom Enums

# List of encoding types
class Encoding(Enum):


    LATIN_1 = "latin1"

# List of job processing types
class JobType(Enum):


    BATCH = "BATCH"  # No comma between two constants otherwise the output will be tuple not value
    SINGLE = "SINGLE"

# List of regular expressions
class RegexFilter(Enum):


    NON_ALPHA_NUMERIC = "[^\w\s]"  # ^ NOT, \w WORDS, \s WHITESPACE
    DIGITS_ONLY = "[\d]"  # \d DIGITS
    SQUARE_BRACKET = "\\[|\\]"  # | AND, \[ left squre bracket,\] right square bracket
    SINGLE_QUOTATION = "\\'"  # \' QUOTATION 
    SQUARE_BRACKET_AND_SINGLE_QUOTATION = "\\[|\\]|\\'"  # | AND, \[ left squre bracket,\] right square bracket
    SINGLE_COMMA = ","  # represent single comma

# KV list of languages and ISO code
class Language(Enum):


    ENGLISH = "en_US"

# Column Name 
class JobId(Enum):


    JOB_ID = "job_id"

# Orient parameters
class OrientPara(Enum):


    COLUMN = "columns"
    INDEX = "index"

# Columns name 
class ColumnName(Enum):


    TITLE = "title"
    TEXTDATA = "textData"
    LABEL = "label"
    PROBABILITIES = "probabilities"
    JOBID = "job_id"
    TIME = "time"
    UPDATED = "updated_labels"
    PREDID = "prediction_id"
    UPDATEDLABEL = "updated_labels"
    RECORDID = "record_id"

# Encoding type
class EncodingType(Enum):


    UTF8 = "utf-8"

# Label type
class LabelType(Enum):  


    HAZARD = "hazard"
    THEME = "theme"
