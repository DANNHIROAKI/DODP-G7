from libs import * # Importing all the libraries


# PART ONE: Preprocessing Book's Data
def Preprocess(books_path, ratings_path, users_path):
    print("PART ONE: Preprocessing Book's Data")
    matplotlib.use('Agg')  # Avoid GUI Contradications
    warnings.filterwarnings(
        "ignore", 
        category=DeprecationWarning
    )
    ratings, books, users = Load_data(books_path, ratings_path, users_path) # 1.
    users_cleaned = Preprocess_users(users) # 2.
    books_cleaned = Preprocess_books(books) # 3.
    ratings_cleaned = Preprocess_ratings(ratings) # 4.
    merged_data = Preprocess_final(
        ratings_cleaned, 
        books_cleaned, 
        users_cleaned
    ) # 5.
    merged_data.to_csv('Data-Files\\BX-Preprocessed.csv', index=False)
    print(merged_data.head())
    return merged_data

# 1. Data Loading
def Load_data(books_path, ratings_path, users_path):
    print("Preprocessing # 1. Load the data from the CSV files")
    books = pd.read_csv(books_path)
    ratings = pd.read_csv(ratings_path)
    users = pd.read_csv(users_path)
    return ratings, books, users

# 2. Preprocessing Users
def Preprocess_users(users):
    print("Preprocessing # 2. Preprocessing Users")
    Preprocess_users_location(users) # 2.1.
    Preprocess_users_age(users) # 2.2.
    users_cleaned = Preprocess_users_final(users) # 2.3.
    return users_cleaned

# 2.1. Preprocessing User-Location
def Preprocess_users_location(users):
    print("Preprocessing # 2.1. Preprocessing User-Location")
    Preprocess_users_location_city(users) # 2.1.1.
    Preprocess_users_location_country(users) # 2.1.2.
    Preprocess_users_location_state(users) # 2.1.3.
    Preprocess_users_location_encoder(users) # 2.1.4.

# 2.1.1. Preprocessing User-City
def Preprocess_users_location_city(users):
    print("Preprocessing # 2.1.1. Preprocessing User-City")
    users.drop(['User-City'], axis=1, inplace=True)

# 2.1.2. Preprocessing User-Country
def Preprocess_users_location_country(users):
    Preprocess_users_location_country_clean(users) # 2.1.2.1.
    Preprocess_users_location_country_fillna(users) #2.1.2.2.

# 2.1.2.1. Cleaning of User-Country non-missing values
def Preprocess_users_location_country_clean(users):
    print("Preprocessing # 2.1.2.1. Cleaning of User-Country non-missing values")
    users["User-Country"] = users["User-Country"].str.replace('"', ' ').str.strip().str.upper().replace(' ', pd.NA)

# 2.1.2.2. Filling in missing values for User-Country
def Preprocess_users_location_country_fillna(users):
    print("Preprocessing # 2.1.2.2. Filling in missing values for User-Country")
    state_to_country = Preprocess_users_location_country_fillna_map(users) # 2.1.2.2.1.
    Preprocess_users_location_country_fillna_fill(users, state_to_country) # 2.1.2.2.2.

# 2.1.2.2.1. Create a mapping of State⬅️➡️Country
def Preprocess_users_location_country_fillna_map(users):
    print("Preprocessing # 2.1.2.2.1. Create a mapping of State——Country")
    valid_countries = users.dropna(subset=['User-Country'])
    country_counts = valid_countries.groupby('User-State')['User-Country'].value_counts().sort_values(ascending=False)
    state_to_country = {}
    for state, country_freq in country_counts.groupby(level=0):
        state_to_country[state] = country_freq.idxmax()[1]
    return state_to_country

# 2.1.2.2.2. Fill in missing Country values based on mapping
def Preprocess_users_location_country_fillna_fill(users, state_to_country):
    print("Preprocessing # 2.1.2.2.2. Fill in missing Country values based on mapping")
    for index, row in users.iterrows():
        if pd.isna(row['User-Country']) and pd.notna(row['User-State']):
            if row['User-State'] in state_to_country:
                users.at[index, 'User-Country'] = state_to_country[row['User-State']]
    users.dropna(subset=['User-Country'], inplace=True)

# 2.1.3. Preprocssing User-States
def Preprocess_users_location_state(users):
    print("Preprocessing # 2.1.3. Preprocssing User-States")
    Preprocess_users_location_state_clean(users) # 2.1.3.1.
    valid_states_per_country = Preprocess_users_location_state_percentage(users) #2.1.3.2.
    Preprocess_users_location_state_generate(users, valid_states_per_country) #2.3.2.2.
    Preprocess_users_location_state_top100(users) #2.1.3.4.

# 2.1.3.1. Cleaning of User-State non-missing values
def Preprocess_users_location_state_clean(users):
    print("Preprocessing # 2.1.3.1. Cleaning of User-State non-missing values")
    users["User-State"] = users["User-State"].str.replace('"', ' ').str.strip().str.lower().replace(' ', pd.NA)

# 2.1.3.2. Calculate the percentage of users in each state / country
def Preprocess_users_location_state_percentage(users):
    print("Preprocessing # 2.1.3.2. Calculate the percentage of users in each state / country")
    total_users = len(users)
    valid_states_per_country = {}
    for country in users['User-Country'].unique():
        country_users = users[users['User-Country'] == country]
        state_counts = country_users['User-State'].value_counts()
        state_ratios = (state_counts / total_users * 100)
        valid_states = state_ratios[state_ratios > 0.5]
        valid_states_per_country[country] = valid_states.to_dict()
    return valid_states_per_country

# 2.1.3.3. Generate User-Location columns based on the logic of the flowchart
def Preprocess_users_location_state_generate(users, valid_states_per_country):
    print("Preprocessing # 2.1.3.3. Generate User-Location columns based on the logic of the flowchart") 
    def determine_location(row):
        country = row['User-Country']
        state = row['User-State']
        valid_states = valid_states_per_country.get(country, []) 
        if len(valid_states) == 0:
            return f"{country}(entire)" 
        elif state and state in valid_states:
            return f"{country}({state})"  
        else:
            return f"{country}(others)"  
    users['User-Location'] = users.apply(determine_location, axis=1)

# 2.1.3.4. Select the top 100 User-Locations
def Preprocess_users_location_state_top100(users):
    print("Preprocessing # 2.1.3.4. Select the top 100 User-Locations")
    top_100_locations = users['User-Location'].value_counts().head(100).index
    users = users[users['User-Location'].isin(top_100_locations)]
    users = users.drop(columns=['User-Country', 'User-State'])

# 2.1.4. Encoding: Converting the Location text into numerical values
def Preprocess_users_location_encoder(users):
    print("Preprocessing # 2.1.4. Encoding: Converting the Location text into numerical values")
    ordinal_encoder = OrdinalEncoder()
    users['User-Location-Encoded'] = ordinal_encoder.fit_transform(users[['User-Location']])

# 2.2. Preprocessing User-Age
def Preprocess_users_age(users):
    print("Preprocessing # 2.2. Preprocessing User-Age")
    Preprocess_users_age_clean(users) # 2.2.1.
    Preprocess_users_age_fillna(users) # 2.2.2.
    Preprocess_users_age_outliers(users) # 2.2.3.
    Preprocess_users_age_binning(users) # 2.2.4.


# 2.2.1. Cleaning of User-Age non-missing values
def Preprocess_users_age_clean(users):
    print("Preprocessing # 2.2.1. Cleaning of User-Age non-missing values")
    users["User-Age"] = users["User-Age"].str.replace('"', "").replace("NaN", pd.NA).astype(float).astype("Int64")

# 2.2.2. Filling in missing values for User-Age
def Preprocess_users_age_fillna(users):
    print("Preprocessing # 2.2.2. Filling in missing values for User-Age")
    mean_age = users['User-Age'].mean(skipna=True)
    std_age = users['User-Age'].std(skipna=True)
    np.random.seed(42) 
    users['User-Age'] = users['User-Age'].apply(lambda x: np.random.normal(mean_age, std_age) if pd.isna(x) else x)

# 2.2.3. Handling outliers in User-Age
def Preprocess_users_age_outliers(users):
    print("Preprocessing # 2.2.3. Handling outliers in User-Age")
    users['User-Age'] = users['User-Age'].apply(lambda x: 99 if x >= 100 else (1 if x <= 0 else x))

# 2.2.4. Binning User-Age
def Preprocess_users_age_binning(users):
    print("Preprocessing # 2.2.4. Binning User-Age")
    bins = list(range(0, 101, 5))  
    labels = list(range(1, 21))  
    users["User-Age-Binned"] = pd.cut(
        users["User-Age"], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
# 2.3. Merge & Scaling
def Preprocess_users_final(users):
    scaler = StandardScaler()
    users[['User-Age-Binned', 'User-Location-Encoded']] = scaler.fit_transform(users[[
        'User-Age-Binned', 
        'User-Location-Encoded'
    ]])
    users_cleaned = users.drop(columns=['User-Location', 'User-Age'])
    return users_cleaned

# 3. Preprocessing Books
def Preprocess_books(books):
    print("Preprocessing # 3. Preprocessing Books")
    Preprocess_books_publication(books)  # 3.1.
    Encode_books_PublisherAuthor(books) # 3.2.
    Preprocess_books_title(books)  # 3.3.
    books_cleaned = Preprocess_books_final(books)# 3.4.  
    return books_cleaned


# 3.1. Preprocessing Year-Of-Publication
def Preprocess_books_publication(books):
    print("Preprocessing # 3.1. Preprocessing Year-Of-Publication")
    Preprocess_books_publication_outliers(books) # 3.1.1.
    Preprocess_books_publication_binning(books) # 3.1.2.
    Preprocess_books_publication_encoder(books) # 3.1.3.

# 3.1.1. Handling of Publication Outliers
def Preprocess_books_publication_outliers(books):
    print("Preprocessing # 3.1.1. Handling of Publication Outliers")
    books["Year-Of-Publication"] = books["Year-Of-Publication"].clip(upper=2024, lower=1930)
    
# 3.1.2. Binning Year-Of-Publication
def Preprocess_books_publication_binning(books):
    print("Preprocessing # 3.1.2. Binning Year-Of-Publication")
    bins = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
    books["Year-Of-Publication-Binned"] = pd.cut(books["Year-Of-Publication"], bins=bins, right=False)
    
# 3.1.3. Encoder Year-Of-Publication-Binned
def Preprocess_books_publication_encoder(books):
    print("Preprocessing # 3.1.3. Encoder Year-Of-Publication-Binned")
    ordinal_encoder = OrdinalEncoder()
    encoded_values = ordinal_encoder.fit_transform(books['Year-Of-Publication-Binned'].values.reshape(-1, 1))
    books['Year-Of-Publication-Binned'] = encoded_values

# 3.2. Encoding Publisher & Author
def Encode_books_PublisherAuthor(books):
    print("Preprocessing # 3.2. Encoding Publisher & Author")
    ordinal_encoder = OrdinalEncoder()
    books['Book-Author-Encoded'] = ordinal_encoder.fit_transform(books['Book-Author'].values.reshape(-1, 1))
    books['Book-Publisher-Encoded'] = ordinal_encoder.fit_transform(books['Book-Publisher'].values.reshape(-1, 1))

# 3.3. Book-Title Preprocessing
def Preprocess_books_title(books):
    print("Preprocessing # 3.3. Book-Title Preprocessing")
    Preprocess_books_title_clean(books) # 3.3.1. 
    Preprocess_books_title_wordcloud(books) # 3.3.2. 
    Preprocess_books_title_vector(books) # 3.3.3. 
    Preprocess_books_title_DimensionReduct(books) # 3.3.4. 

# 3.3.1. Cleaning the titles
def Preprocess_books_title_clean(books):
    nlp = spacy.load('en_core_web_sm')
    def Preprocess_text_spacy(text):
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        clean_text = ' '.join(tokens)
        return clean_text
    tqdm.pandas(desc="Preprocessing # 3.3.1. Cleaning the titles (To create word cloud)") 
    books['Book-Title'] = books['Book-Title'].progress_apply(Preprocess_text_spacy)

# 3.3.2. Create Word Cloud
def Preprocess_books_title_wordcloud(books):
    print("Preprocessing # 3.3.2. Create Word Cloud (In ./Plots)")
    wordcloud = WordCloud(
        width=1800, 
        height=1400, 
        background_color='white',
        font_path='Fonts/arial.TTF'  
    ).generate(' ' .join(books['Book-Title']))
    plt.figure(figsize=(15, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.savefig('Plots\\wordcloud.png')
    plt.close()

# 3.3.3. Word➡️Vector: BERT based method
def Preprocess_books_title_vector(books):
    print("Preprocessing # 3.3.3. Word to Vector: BERT based method")
    tokenizer, model = Preprocess_books_title_vector_loadmodel() # 3.3.3.1. 
    model, device = Preprocess_books_title_vector_GPU(model) # 3.3.3.2. 
    Preprocess_books_title_vector_batch(
        books, 
        tokenizer, 
        model, 
        device
    ) # 3.3.3.4. 
    
# 3.3.3.1. Load pretrained model
def Preprocess_books_title_vector_loadmodel():
    print("Preprocessing # 3.3.3.1. Load pretrained model")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

# 3.3.3.2. Trying to move the pretrained model to GPU
def Preprocess_books_title_vector_GPU(model):
    print("Preprocessing # 3.3.3.2. Trying to move the pretrained model to GPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

# 3.3.3.3. Batch text processing
def Preprocess_books_title_vector_batch(
        books, 
        tokenizer, 
        model, 
        device
    ):
    print("Preprocessing # 3.3.3.3. Batch text processing")
    batch_size = 32
    all_embeddings = []
    total_batches = (len(books['Book-Title']) + batch_size - 1) // batch_size
    progress_bar = tqdm(
        total=total_batches, 
        desc="Preprocessing # 3.3.3.4. Training BERT Model"
    )
    for i in range(0, len(books['Book-Title']), batch_size):
        batch_titles = books['Book-Title'][i:i+batch_size].tolist()
        embeddings = Preprocess_books_title_vector_train(
            tokenizer, 
            model, 
            batch_titles, 
            device, progress_bar
        )
        all_embeddings.extend(embeddings.numpy())
    books['Book-Title-Embeddings'] = all_embeddings
    progress_bar.close()

# 3.3.3.4. Training BERT Model
def Preprocess_books_title_vector_train(
        tokenizer, 
        model, 
        texts, 
        device, 
        progress_bar
    ):
    encoded_input = tokenizer(
        texts, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=64
    ).to(device)
    with torch.no_grad():
        output = model(**encoded_input)
    sentence_embeddings = torch.mean(
        output.last_hidden_state, 
        dim=1
    )
    progress_bar.update(1)
    return sentence_embeddings.cpu()

# 3.3.4. Vector Dimension Reduction: UMAP Baesd
def Preprocess_books_title_DimensionReduct(books):
    print("Preprocessing # 3.3.4. UMAP Baesd Vector Dimension Reduction")
    reducer = umap.UMAP(
        n_neighbors=15, 
        n_components=1, 
        metric='cosine'
    )
    bert_embeddings = np.array(books['Book-Title-Embeddings'].tolist())
    embedding = reducer.fit_transform(bert_embeddings)
    embedding_values = [value[0] for value in embedding]
    embedding_values = np.array(embedding_values)
    books['Book-Title-Embeddings'] = embedding_values

# 3.4. Merge Book Data
def Preprocess_books_final(books):
    print("Preprocessing # 3.4. Merge Book Data")
    columns_to_keep = [
        'ISBN', 
        'Year-Of-Publication', 
        'Year-Of-Publication-Binned', 
        'Book-Author-Encoded', 
        'Book-Publisher-Encoded',
        'Book-Title-Embeddings'
    ]
    for column in columns_to_keep:
        if column not in books.columns:
            raise ValueError(f"Column {column} is not present in the DataFrame")
    books_cleaned = books[columns_to_keep]
    return books_cleaned
    
# 4. Preprocessing Ratings
def Preprocess_ratings(ratings):
    print("Preprocessing # 4. Preprocessing Ratings")
    ratings.loc[ratings['Book-Rating'] <= 5, 'Book-Rating'] = 5
    ratings_cleaned = ratings
    return ratings_cleaned

# 5. For All the Data: Merge & Review
def Preprocess_final(
        ratings_cleaned, 
        books_cleaned, 
        users_cleaned
    ):
    print("Preprocessing # 5. For All the Data: Merge & Review")
    merged_data = Preprocess_final_merge(
        ratings_cleaned, 
        books_cleaned, 
        users_cleaned
    ) # 5.1.
    Preprocess_final_correlation(merged_data) # 5.2.
    return merged_data


# 5.1. Merge
def Preprocess_final_merge(
        ratings_cleaned, 
        books_cleaned, 
        users_cleaned
    ):
    print("Preprocessing # 5.1. Merge")
    merged_data = pd.merge(
        ratings_cleaned, 
        books_cleaned, 
        on='ISBN', 
        how='inner'
    )
    merged_data = pd.merge(
        merged_data, 
        users_cleaned, 
        on='User-ID', 
        how='inner'
    )
    merged_data = merged_data[[
        'ISBN',
        'Book-Rating', 
        'Year-Of-Publication-Binned',
        'Book-Author-Encoded', 
        'Book-Publisher-Encoded', 
        'Book-Title-Embeddings',
        'User-Age-Binned', 
        'User-Location-Encoded',
        'User-ID'
    ]]
    return merged_data

# 5.2. correlation
def Preprocess_final_correlation(merged_data):
    print("Preprocessing # 5.2. Correlation (In ./Plots)")
    pearson_corr_matrix = Preprocess_final_correlation_pearson(merged_data)  # 5.2.1.
    mi_matrix = Preprocess_final_correlation_mutualinfo(merged_data)  # 5.2.2.
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize=(12, 5)
    )
    sns.heatmap(
        pearson_corr_matrix, 
        ax=axes[0], 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f"
    )
    axes[0].set_title('Linear relationship correlation (Pearson)')
    sns.heatmap(
        mi_matrix,
        ax=axes[1], 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f"
    )
    axes[1].set_title('Non-linear relationship correlation (Mutual Information)')
    axes[0].set_ylabel('Features')  
    axes[1].set_ylabel('')  
    axes[0].set_xlabel('Features')
    axes[1].set_xlabel('Features')
    plt.setp(
        axes[0].get_yticklabels(), 
        rotation=0
    )  
    plt.setp(
        axes[0].get_xticklabels(), 
        rotation=90
    ) 
    plt.setp(
        axes[1].get_xticklabels(), 
        rotation=90
    ) 
    plt.tight_layout()
    plt.savefig('Plots\\correlation.png')
    plt.close(fig)

# 5.2.1. Linear relationship correlation (Pearson)
def Preprocess_final_correlation_pearson(merged_data):
    merged_data = merged_data.drop(columns=['ISBN'])
    print("Preprocessing # 5.2.1. Linear relationship correlation (Pearson)")
    return merged_data.corr()


# 5.2.2. Non-linear relationship correlation (Mutual Information)
def Preprocess_final_correlation_mutualinfo(merged_data):
    warnings.filterwarnings("ignore")
    merged_data = merged_data.drop(columns=['ISBN'])
    print("Preprocessing # 5.2.2. Non-linear relationship correlation (Mutual Information)")
    X = merged_data
    n_features = X.shape[1]
    mi_matrix = pd.DataFrame(
        data=np.zeros((n_features, n_features)), 
        columns=X.columns, 
        index=X.columns
    )
    for i in range(n_features):
        for j in range(i + 1, n_features):
            x = X.iloc[:, i].values.reshape(-1, 1)
            y = X.iloc[:, j].values
            mi = mutual_info_regression(x, y)
            mi_matrix.iloc[i, j] = mi[0]
            mi_matrix.iloc[j, i] = mi[0]
    return mi_matrix





