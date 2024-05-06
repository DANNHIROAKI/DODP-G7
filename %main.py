from preproccessing import Preprocess
from preproccessing_newbooks import PreprocessNewbook
from classification import Classification
from recommendation import Recommendation
import pandas as pd

# PART ONE: Preprocessing Book's Data
data = Preprocess(
    "Data-Files\\BX-Books.csv", 
    "Data-Files\\BX-Ratings.csv", 
    "Data-Files\\BX-Users.csv"
)
# PART TWO: Classification
data_classification = data.drop(columns=['ISBN', 'User-ID'])
Classification(data_classification)
# PERT THREE: Preprocessing NewBook's Data
newdata = PreprocessNewbook(
    "Data-Files\\BX-NewBooks.csv", 
    "Data-Files\\BX-NewBooksRatings.csv", 
    "Data-Files\\BX-NewBooksUsers.csv"
)
# PART FOUR: Recommendation System
Recommendation(
    data.sample(n=20000, random_state=42), 
    newdata.sample(n=20000, random_state=42)
)


# ################# Fast execution Version ####################
# ### PART ONE: Preprocessing Book's Data
# data = (pd.read_csv('Data-Files\\BX-Preprocessed.csv'))
# ### PART TWO: Classification
# Classification(data.drop(columns=['ISBN']).sample(n=180000, random_state=42))
# ### PERT THREE: Preprocessing NewBook's Data
# newdata = (pd.read_csv('Data-Files\\BX-NewPreprocessed.csv'))
# ### PART FOUR: Recommendation System
# Recommendation(
#     data.sample(n=20000, random_state=42), 
#     newdata.sample(n=20000, random_state=42)
# )