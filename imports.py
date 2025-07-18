import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, jsonify
import os
import pickle
import joblib
import pyarabic.araby as araby
from camel_tools.utils.normalize import normalize_unicode
import re
import sklearn_crfsuite
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from itertools import combinations
