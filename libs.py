# Standard libraries
import io
import os
import time
import warnings
import contextlib

# Data processing libraries
import numpy as np
import pandas as pd

# Natural Language Processing libraries
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Machine Learning and Model Evaluation libraries
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix,
    mutual_info_score, 
)
from sklearn.feature_selection import mutual_info_regression

# Visualization libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Miscellaneous
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import umap

# Collaborative Filtering and Recommendation Systems
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split
from surprise.accuracy import rmse

# Suppress warnings for better clarity
warnings.filterwarnings('ignore')
