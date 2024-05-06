from libs import * # Importing all the libraries


# PART FOUR: RECOMMENDATION SYSTEM
def Recommendation(train_file, test_file):
    print("PART FOUR: Recommendation System")
    Recommendation_ml(train_file, test_file) # 1. 
    Recommendation_dl(train_file, test_file) # 2.


# 1. Machine Learning Method
def Recommendation_ml(train_file, test_file):
    print("Recommendation # 1. Machine Learning Method")
    trainset, testset, train_data = Recommendation_ml_data(
        train_file, test_file, 
        sample_size=10000, random_state=42
    ) # 1.1. 
    Recommendation_ml_knn(trainset, testset, train_data) # 1.2.
    Recommendation_ml_svd(trainset, testset, train_data) # 1.3.



# 1.1. Data Loading and Preparation
def Recommendation_ml_data(
        train_file, test_file, 
        sample_size=10000, random_state=42
    ):
    print("Recommendation # 1.1. Data Loading and Preparation")
    reader = Reader(rating_scale=(1, 10))
    train_X_sample = train_file.sample(
        n=sample_size, 
        random_state=random_state
    )
    test_X_sample = test_file.sample(
        n=sample_size, 
        random_state=random_state
    )
    train_data = Dataset.load_from_df(
        train_X_sample[['User-ID', 'ISBN', 'Book-Rating']], 
        reader
    )
    test_data = Dataset.load_from_df(
        test_X_sample[['User-ID', 'ISBN', 'Book-Rating']], 
        reader
    )
    trainset = train_data.build_full_trainset()
    testset = test_data.build_full_trainset().build_testset()
    return trainset, testset, train_data

# 1.2. KNN Based Model
def Recommendation_ml_knn(trainset, testset, train_data): 
    print("Recommendation # 1.2. KNN Based Model")
    # 1.2.1.
    print("Recommendation # 1.2.1. K Parameter Optimization")
    k_values_user, results_user, optimal_k_user, sim_options_user = Recommendation_ml_knn_k(
        train_data, 
        user_based=True
    )
    k_values_item, results_item, optimal_k_item, sim_options_item = Recommendation_ml_knn_k(
        train_data, 
        user_based=False
    )
    Recommendation_ml_knn_plot(
        k_values_user, 
        results_user, 
        k_values_item, 
        results_item
    ) # 1.2.2. 
    print("Recommendation # 1.2.3. RMSE After Optimized") # 1.2.3. 
    final_rmse_user = Recommendation_ml_rmse(
        trainset, 
        testset, 
        optimal_k_user, 
        sim_options_user
    )
    print(
        "Recommendation # 1.2.3.1. Optimized User-Based KNN RMSE: ", 
        final_rmse_user
    )
    final_rmse_item = Recommendation_ml_rmse(
        trainset, 
        testset, 
        optimal_k_item, 
        sim_options_item
    )
    print(
        "Recommendation # 1.2.3.2. Optimized Item-Based KNN RMSE: ", 
        final_rmse_item
    )



# 1.2.1. K Parameter Optimization
def Recommendation_ml_knn_k(train_data, user_based=True):
    with contextlib.redirect_stdout(io.StringIO()):
        k_values = np.arange(1, 101)
        results = []
        for k in tqdm(
            k_values, desc=
                "Recommendation # 1.2.1.1. Training User-Based KNN" 
                if user_based 
                else 
                "Recommendation # 1.2.1.2. Training Item-Based KNN"
        ):
            sim_options = {'name': 'msd', 'user_based': user_based}
            model = KNNBasic(k=k, sim_options=sim_options)
            cv_results = cross_validate(
                model, 
                train_data, 
                measures=['RMSE'], 
                cv=5, 
                verbose=False
            )
            mean_rmse = np.mean(cv_results['test_rmse'])
            results.append(mean_rmse)
        optimal_k = k_values[np.argmin(results)]
    return k_values, results, optimal_k, sim_options


# 1.2.2. Visualization of RMSE vs. k Values
import matplotlib.pyplot as plt
def Recommendation_ml_knn_plot(
        k_values_user, 
        results_user, 
        k_values_item, 
        results_item
    ):
    print("Recommendation # 1.2.2. Visualization of RMSE vs. k Values (In ./Plots_NewBook)")
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    plt.plot(
        k_values_user, 
        results_user, 
        marker='o'
    )
    plt.title('User-Based KNN', fontsize=16)  
    plt.xlabel('Number of Neighbors (k)', fontsize=14)
    plt.ylabel('Average RMSE', fontsize=14)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(
        k_values_item, 
        results_item, 
        marker='o'
    )
    plt.title('Item-Based KNN', fontsize=16)
    plt.xlabel('Number of Neighbors (k)', fontsize=14)
    plt.ylabel('Average RMSE', fontsize=14)
    plt.grid(True)
    plt.suptitle('RMSE vs. k values', fontsize=18)
    plt.savefig('Plots_NewBook/RMSE_vs_k_values.png')
    plt.close()


# 1.2.3. KNN's RMSE After Optimized
def Recommendation_ml_rmse(
        trainset, testset, 
        optimal_k, 
        sim_options
    ):
    with contextlib.redirect_stdout(io.StringIO()):
        model = KNNBasic(
            k=optimal_k, 
            sim_options=sim_options
        )
        model.fit(trainset)
        predictions = model.test(testset)
        final_rmse = accuracy.rmse(predictions, verbose=False)
    return final_rmse


# 1.3. SVD Based Model
def Recommendation_ml_svd(trainset, testset, train_data):
    gs = Recommendation_ml_svd_cv(train_data)
    optimized_rmse = Recommendation_ml_svd_best(trainset, testset, gs)
    print(f"Optimized RMSE: {optimized_rmse}")
    Recommendation_ml_svd_plot(train_data)


# 1.3.1. Finding Optimal Parameters with Cross-Validation
def Recommendation_ml_svd_cv(train_data):
    print("Recommendation # 1.3.1. Finding Optimal Parameters with Cross-Validation")
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30, 40],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(train_data)
    return gs

# 1.3.2. Retrain Model with Best Parameters
def Recommendation_ml_svd_best(trainset, testset, gs):
    best_svd = SVD(
        n_factors=gs.best_params['rmse']['n_factors'],
        n_epochs=gs.best_params['rmse']['n_epochs'],
        lr_all=gs.best_params['rmse']['lr_all'],
        reg_all=gs.best_params['rmse']['reg_all']
    )
    best_svd.fit(trainset)
    predictions = best_svd.test(testset)
    optimized_rmse = accuracy.rmse(predictions, verbose=False)  
    print(f"Recommendation # 1.3.2. Retrain Model with Best Parameters - Final RMSE: {optimized_rmse:.6f}")  
    return optimized_rmse


# 1.3.3. Visualizing RMSE vs. Number of Factors
def Recommendation_ml_svd_plot(train_data):
    print("Recommendation # 1.3.3. Visualizing RMSE vs. Number of Factors (In ./Plots_NewBook)")
    n_factors_values = np.arange(1, 101)  # 改为从1到100
    results = []
    for n_factors in n_factors_values:
        model = SVD(
            n_factors=n_factors, 
            n_epochs=20, 
            lr_all=0.005, 
            reg_all=0.02
        )
        cv_results = cross_validate(
            model, 
            train_data, 
            measures=['RMSE'], 
            cv=3, 
            verbose=False
        )
        mean_rmse = np.mean(cv_results['test_rmse'])
        results.append(mean_rmse)
    plt.figure(figsize=(10, 8))
    plt.plot(
        n_factors_values, 
        results, 
        marker='o', 
        linestyle='-'
    )
    plt.title('RMSE vs. Number of Factors', fontsize=14)
    plt.xlabel('Number of N_Factors', fontsize=12)
    plt.ylabel('Average RMSE', fontsize=12)
    plt.grid(True)
    plt.savefig('Plots_NewBook/RMSE_vs_NFactors_values.png')
    plt.close()



# 2. Deep Learning Method (Dense Layers NN)
def Recommendation_dl(train_file, test_file):
    warnings.filterwarnings('ignore')
    print("Recommendation # 2. Deep Learning Method")
    X_train, y_train, X_test, y_test = Recommendation_dl_data(train_file, test_file)
    model = Recommendation_dl_model(X_train.shape[1])
    history = Recommendation_dl_train(model, X_train, y_train, X_test, y_test)
    Recommendation_dl_plot(history)


# 2.1. Data Loading and Preparation for Deep Learning
def Recommendation_dl_data(train_file, test_file):
    warnings.filterwarnings('ignore')
    print("Recommendation # 2.1. Data Loading and Preparation for Deep Learning")
    train_df = train_file
    test_df = test_file
    X_train = train_df[['Book-Author-Encoded', 'Book-Publisher-Encoded', 'Book-Title-Embeddings']].values
    X_test = test_df[['Book-Author-Encoded', 'Book-Publisher-Encoded', 'Book-Title-Embeddings']].values
    y_train = train_df['Book-Rating'].values.astype('float32')
    y_test = test_df['Book-Rating'].values.astype('float32')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

# 2.2. Training and Evaluating the Final Model
def Recommendation_dl_model(input_shape):
    warnings.filterwarnings('ignore')
    print("Recommendation # 2.2. Training and Evaluating the Final Model")
    model = Sequential([
        Dense(128, input_shape=(input_shape,)),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        Dense(32),
        LeakyReLU(alpha=0.01),
        Dense(1)
    ])
    rmsprop_optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-7, momentum=0.0)
    def rmse(y_true, y_pred):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))
    model.compile(optimizer=rmsprop_optimizer, loss='mse', metrics=[rmse])
    return model

# 2.3. Training and Evaluating the Final Model
def Recommendation_dl_train(model, X_train, y_train, X_test, y_test):
    warnings.filterwarnings('ignore')
    print("Recommendation # 2.3. Training and Evaluating the Final Model")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=0.01, mode='min', min_lr=1e-6)
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        callbacks=[reduce_lr]
    )
    test_loss, test_rmse = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test RMSE: {test_rmse}")
    return history

# 2.4. Visualization of Training and Validation RMSE
def Recommendation_dl_plot(history):
    warnings.filterwarnings('ignore')
    print("Recommendation # 2.4. Visualization of Training and Validation RMSE (In ./Plots_NewBook)")
    train_rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']
    epochs = range(1, len(train_rmse) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, 'bo-', label='Training RMSE')
    plt.plot(epochs, val_rmse, 'ro-', label='Validation RMSE')
    plt.title('Training and Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.savefig('Plots_NewBook/Training_and_Validation_RMSE.png')
    plt.close()


