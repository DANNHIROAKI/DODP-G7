from libs import * # Importing all the libraries

# PART TWO: Classification
def Classification(data):
    print("PART TWO: Classification")

    DT_data, RF_data, KNN_data, GBM_data = classification_load(data) # 1. 
    df_train, rf_train, knn_train, gbm_train = classification_train_models(
        *DT_data[:4], 
        *RF_data[:4], 
        *KNN_data[:4], 
        *GBM_data[:4]
    ) # 2.
    classification_performance(
        df_train[2], df_train[1], 
        rf_train[2], rf_train[1], 
        knn_train[2], knn_train[1], 
        gbm_train[2], gbm_train[1]
    ) # 3. 
    classification_importance(
        df_train[0], rf_train[0], knn_train[0], gbm_train[0],
        DT_data[4], RF_data[4], GBM_data[4], 
        KNN_data[1], knn_train[2], 
        knn_train[1]
    ) # 4.

# 1. Load and segment data for models
def classification_load(data):
    print("Classification # 1. Load and segment data for models")
    X_DT_train, X_DT_test, y_DT_train, y_DT_test, X_DT, y_DT = classification_load_dt(data)  # 1.1.
    X_RF_train, X_RF_test, y_RF_train, y_RF_test, X_RF, y_RF = classification_load_rf(data)  # 1.2.
    X_KNN_train, X_KNN_test, y_KNN_train, y_KNN_test, X_KNN, y_KNN = classification_load_knn(data)  # 1.3.
    X_GBM_train, X_GBM_test, y_GBM_train, y_GBM_test, X_GBM, y_GBM = classification_load_gbm(data)  # 1.4. 
    DT_data = (
        X_DT_train, X_DT_test, 
        y_DT_train, y_DT_test, 
        X_DT, y_DT
    )
    RF_data = (
        X_RF_train, X_RF_test, 
        y_RF_train, y_RF_test, 
        X_RF, y_RF
    )
    KNN_data = (
        X_KNN_train, X_KNN_test, 
        y_KNN_train, y_KNN_test, 
        X_KNN, y_KNN
    )
    GBM_data = (
        X_GBM_train, X_GBM_test, 
        y_GBM_train, y_GBM_test, 
        X_GBM, y_GBM
    )
    return DT_data, RF_data, KNN_data, GBM_data

# 1.1. Load data for Decision Tree
def classification_load_dt(data):
    print("Classification # 1.1. Load data for Decision Tree")
    X_DT = data.drop(columns=['Book-Rating'])
    y_DT = data['Book-Rating']
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_DT))
    test_set_size = int(len(X_DT) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_DT_train = X_DT.iloc[train_indices]
    X_DT_test = X_DT.iloc[test_indices]
    y_DT_train = y_DT.iloc[train_indices]
    y_DT_test = y_DT.iloc[test_indices]
    return X_DT_train, X_DT_test, y_DT_train, y_DT_test, X_DT, y_DT

# 1.2. Load data for Random Forest
def classification_load_rf(data):
    print("Classification # 1.2. Load data for Random Forest")
    X_RF = data.drop(columns=['Book-Rating'])
    y_RF = data['Book-Rating']
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_RF))
    test_set_size = int(len(X_RF) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_RF_train = X_RF.iloc[train_indices]
    X_RF_test = X_RF.iloc[test_indices]
    y_RF_train = y_RF.iloc[train_indices]
    y_RF_test = y_RF.iloc[test_indices]
    return X_RF_train, X_RF_test, y_RF_train, y_RF_test, X_RF, y_RF

# 1.3. Load data for K-Nearest Neighbors
def classification_load_knn(data):
    print("Classification # 1.3. Load data for K-Nearest Neighbors")
    X_KNN = data.drop(columns=['Book-Rating'])
    y_KNN = data['Book-Rating']
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_KNN))
    test_set_size = int(len(X_KNN) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_KNN_train = X_KNN.iloc[train_indices]
    X_KNN_test = X_KNN.iloc[test_indices]
    y_KNN_train = y_KNN.iloc[train_indices]
    y_KNN_test = y_KNN.iloc[test_indices]
    return X_KNN_train, X_KNN_test, y_KNN_train, y_KNN_test, X_KNN, y_KNN

# 1.4. Load data for Gradient Boosting
def classification_load_gbm(data):
    print("Classification # 1.4. Load data for Gradient Boosting")
    X_GBM = data.drop(columns=['Book-Rating'])
    y_GBM = data['Book-Rating']
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_GBM))
    test_set_size = int(len(X_GBM) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_GBM_train = X_GBM.iloc[train_indices]
    X_GBM_test = X_GBM.iloc[test_indices]
    y_GBM_train = y_GBM.iloc[train_indices]
    y_GBM_test = y_GBM.iloc[test_indices]
    return X_GBM_train, X_GBM_test, y_GBM_train, y_GBM_test, X_GBM, y_GBM

# 2. Train the model
def classification_train(
        X_DT_train, X_DT_test, 
        y_DF_train, y_DF_test,
        X_RF_train, X_RF_test, 
        y_RF_train, y_RF_test,
        X_KNN_train, X_KNN_test, 
        y_KNN_train, y_KNN_test,
        X_GBMtrain, X_GBM_test, 
        y_GBM_train, y_GBM_test
    ):
    X_DT_data = X_DT_train, X_DT_test, y_DF_train, y_DF_test
    X_RF_data = X_RF_train, X_RF_test, y_RF_train, y_RF_test
    X_KNN_data = X_KNN_train, X_KNN_test, y_KNN_train, y_KNN_test
    X_GBM_data = X_GBMtrain, X_GBM_test, y_GBM_train, y_GBM_test
    print("Classification # 2. Train the model")
    df_train, rf_train, knn_train, gbm_train = classification_train_models(
        *X_DT_data, 
        *X_RF_data, 
        *X_KNN_data, 
        *X_GBM_data
    )  # 2.1.
    classification_train_result(
        df_train[1], df_train[2], 
        rf_train[1], rf_train[2], 
        knn_train[1], knn_train[2], 
        gbm_train[1], gbm_train[2]
    )  # 2.2.
    return df_train, rf_train, knn_train, gbm_train

# 2.1. Training 4 Classification Models
def classification_train_models(
        X_DT_train, X_DT_test, 
        y_DF_train, y_DF_test,
        X_RF_train, X_RF_test, 
        y_RF_train, y_RF_test,
        X_KNN_train, X_KNN_test, 
        y_KNN_train, y_KNN_test,
        X_GBM_train, X_GBM_test, 
        y_GBM_train, y_GBM_test
    ):
    print("Classification # 2.1. Training 4 Classification Models")
    dt_classifier, dt_predictions, y_DT_test = classification_train_models_dt(
        X_DT_train, X_DT_test, 
        y_DF_train, y_DF_test
    ) # 2.1.1.
    rf_classifier, rf_predictions, y_RF_test = classification_train_models_rf(
        X_RF_train, X_RF_test, 
        y_RF_train, y_RF_test
    ) # 2.1.2.
    knn_classifier, knn_predictions, y_KNN_test = classification_train_models_knn(
        X_KNN_train, X_KNN_test, 
        y_KNN_train, y_KNN_test
    ) # 2.1.3.
    gbm_classifier, gbm_predictions, y_GBM_test = classification_train_models_gbm(
        X_GBM_train, X_GBM_test, 
        y_GBM_train, y_GBM_test
    ) # 2.1.4.
    df_train = (
        dt_classifier, dt_predictions, 
        y_DT_test
    )
    rf_train = (
        rf_classifier, rf_predictions, 
        y_RF_test
    )
    knn_train = (
        knn_classifier, knn_predictions, 
        y_KNN_test
    )
    gbm_train = (
        gbm_classifier, gbm_predictions, 
        y_GBM_test
    )
    return df_train, rf_train, knn_train, gbm_train

# 2.1.1. Training Decision Tree
def classification_train_models_dt(
        X_DT_train, X_DT_test, 
        y_DF_train, y_DF_test
    ):
    print("Classification # 2.1.1. Training Decision Tree")
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_DT_train, y_DF_train)
    dt_predictions = dt_classifier.predict(X_DT_test)
    return dt_classifier, dt_predictions, y_DF_test

# 2.1.2. Training Random Forest
def classification_train_models_rf(
        X_RF_train, X_RF_test, 
        y_RF_train, y_RF_test
    ):
    print("Classification # 2.1.2. Training Random Forest")
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    )
    rf_classifier.fit(X_RF_train, y_RF_train)
    rf_predictions = rf_classifier.predict(X_RF_test)
    return rf_classifier, rf_predictions, y_RF_test

# 2.1.3. Training KNN
def classification_train_models_knn(
        X_KNN_train, X_KNN_test, 
        y_KNN_train, y_KNN_test
    ):
    print("Classification # 2.1.3. Training K-Nearest Neighbors")
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_KNN_train, y_KNN_train)
    knn_predictions = knn_classifier.predict(X_KNN_test)
    return knn_classifier, knn_predictions, y_KNN_test

# 2.1.4. Training GBM
def classification_train_models_gbm(
        X_GBM_train, X_GBM_test, 
        y_GBM_train, y_GBM_test
    ):
    gbm_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    for i in tqdm(range(100), desc="Classification # 2.1.4. Training Gradient Boosting"):
        time.sleep(0.1)
        if i == 0:
            gbm_classifier.fit(X_GBM_train, y_GBM_train)
    gbm_predictions = gbm_classifier.predict(X_GBM_test)
    return gbm_classifier, gbm_predictions, y_GBM_test

# 2.2. Training Result
def classification_train_result(
        dt_predictions, y_DT_test, 
        rf_predictions, y_RF_test, 
        knn_predictions, y_KNN_test, 
        gbm_predictions, y_GBM_test
    ):
    print("Classification # 2.2. Training Result (In ./Plots)")
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    sns.heatmap(
        confusion_matrix(y_DT_test, dt_predictions), 
        annot=True, 
        fmt="d", 
        cmap="Reds", 
        cbar=False
    )
    plt.title("Decision Tree Confusion Matrix")
    plt.subplot(2, 2, 2)
    sns.heatmap(
        confusion_matrix(y_RF_test, rf_predictions), 
        annot=True, 
        fmt="d", 
        cmap="Reds", 
        cbar=False
    )
    plt.title("Random Forest Confusion Matrix")
    plt.subplot(2, 2, 3)
    sns.heatmap(
        confusion_matrix(y_KNN_test, knn_predictions), 
        annot=True, 
        fmt="d", 
        cmap="Reds", 
        cbar=False
    )
    plt.title("KNN Confusion Matrix")
    plt.subplot(2, 2, 4)
    sns.heatmap(
        confusion_matrix(y_GBM_test, gbm_predictions), 
        annot=True, 
        fmt="d", 
        cmap="Reds", 
        cbar=False
    )
    plt.title("GBM Confusion Matrix")
    plt.tight_layout()
    plt.savefig('Plots\\Confusion_Matrix.png')
    plt.close() 
    
# 3. Performance Indicators
def classification_performance(
        y_DT_test, dt_predictions, 
        y_RF_test, rf_predictions, 
        y_KNN_test, knn_predictions, 
        y_GBM_test, gbm_predictions
    ):
    print("Classification # 3. Performance Indicators")
    metrics_DT, metrics_RF, metrics_KNN, metrics_GBM = classification_performance_calculate(
        y_DT_test, dt_predictions, 
        y_RF_test, rf_predictions, 
        y_KNN_test, knn_predictions, 
        y_GBM_test, gbm_predictions
    ) # 3.1.
    classification_performance_store(
        *metrics_DT, 
        *metrics_RF, 
        *metrics_KNN, 
        *metrics_GBM
    ) # 3.2.

# 3.1. Calculate Indicators
def classification_performance_calculate(
        y_DT_test, dt_predictions, 
        y_RF_test, rf_predictions, 
        y_KNN_test, knn_predictions, 
        y_GBM_test, gbm_predictions
    ):
    print("Classification # 3.1. Calculate Indicators") 
    accuracy_DT, f1_DT, classification_rep_DT = classification_performance_calculate_dt(
        y_DT_test, 
        dt_predictions
    ) # 3.1.1.
    accuracy_RF, f1_RF, classification_rep_RF = classification_performance_calculate_rf(
        y_RF_test, 
        rf_predictions
    ) # 3.1.2.
    accuracy_KNN, f1_KNN, classification_rep_KNN = classification_performance_calculate_knn(
        y_KNN_test, 
        knn_predictions
    ) # 3.1.3.
    accuracy_GBM, f1_GBM, classification_rep_GBM = classification_performance_calculate_gbm(
        y_GBM_test, 
        gbm_predictions
    ) # 3.1.4.
    metrics_DT = (
        accuracy_DT, 
        f1_DT, 
        classification_rep_DT
    )
    metrics_RF = (
        accuracy_RF, 
        f1_RF, 
        classification_rep_RF
    )
    metrics_KNN = (
        accuracy_KNN, 
        f1_KNN, 
        classification_rep_KNN
    )
    metrics_GBM = (
        accuracy_GBM, 
        f1_GBM, 
        classification_rep_GBM
    )
    return metrics_DT, metrics_RF, metrics_KNN, metrics_GBM

# 3.1.1. Calculate Decision Tree Performance
def classification_performance_calculate_dt(
        y_DT_test, 
        dt_predictions
    ):
    print("Classification # 3.1.1. Calculate Decision Tree Performance")
    accuracy_DT = accuracy_score(
        y_DT_test, 
        dt_predictions
    )
    f1_DT = f1_score(
        y_DT_test, 
        dt_predictions, 
        average='macro'
    )
    classification_rep_DT = classification_report(
        y_DT_test, 
        dt_predictions
    )
    return accuracy_DT, f1_DT, classification_rep_DT
    
# 3.1.2. Calculate Random Forest Performance
def classification_performance_calculate_rf(
        y_RF_test, 
        rf_predictions
    ):
    print("Classification # 3.1.2. Calculate Random Forest Performance")
    accuracy_RF = accuracy_score(
        y_RF_test, 
        rf_predictions
    )
    f1_RF = f1_score(
        y_RF_test, 
        rf_predictions, 
        average='macro'
    )
    classification_rep_RF = classification_report(
        y_RF_test, 
        rf_predictions
    )
    return accuracy_RF, f1_RF, classification_rep_RF
    
# 3.1.3. Calculate KNN Performance
def classification_performance_calculate_knn(
        y_KNN_test, 
        knn_predictions
    ):
    print("Classification # 3.1.3. Calculate KNN Performance")
    accuracy_KNN = accuracy_score(
        y_KNN_test, 
        knn_predictions
    )
    f1_KNN = f1_score(
        y_KNN_test, 
        knn_predictions, 
        average='macro'
    )
    classification_rep_KNN = classification_report(
        y_KNN_test, 
        knn_predictions
    )
    return accuracy_KNN, f1_KNN, classification_rep_KNN
    
# 3.1.4. Calculate GBM Performance
def classification_performance_calculate_gbm(
        y_GBM_test, 
        gbm_predictions
    ):
    print("Classification # 3.1.4. Calculate GBM Performance")
    accuracy_GBM = accuracy_score(
        y_GBM_test, 
        gbm_predictions
    )
    f1_GBM = f1_score(y_GBM_test, gbm_predictions, average='macro')
    classification_rep_GBM = classification_report(
        y_GBM_test, 
        gbm_predictions
    )
    return accuracy_GBM, f1_GBM, classification_rep_GBM

# 3.2. Store all the Metrics
def classification_performance_store(
        accuracy_DT, f1_DT, classification_rep_DT,
        accuracy_RF, f1_RF, classification_rep_RF,
        accuracy_KNN, f1_KNN, classification_rep_KNN,
        accuracy_GBM, f1_GBM, classification_rep_GBM
    ):
    print("Classification # 3.2. Store all the Metrics")
    classification_performance_store_dt(
        accuracy_DT, f1_DT, classification_rep_DT
    ) # 3.2.1.
    classification_performance_store_rf(
        accuracy_RF, f1_RF, classification_rep_RF
    ) # 3.2.2.
    classification_performance_store_knn(
        accuracy_KNN, f1_KNN, classification_rep_KNN
    ) # 3.2.3.
    classification_performance_store_gbm(
        accuracy_GBM, f1_GBM, classification_rep_GBM
    ) # 3.2.4.

# 3.2.1. Store Decision Tree Metrics
def classification_performance_store_dt(
        accuracy_DT, 
        f1_DT, 
        classification_rep_DT
    ):
    print("Classification # 3.2.1. Store Decision Tree Metrics (In ./Metrics)")
    metrics = {
        'Metric': ['Accuracy', 'F1-score'],
        'Value': [accuracy_DT, f1_DT]}
    df = pd.DataFrame(metrics)
    df.to_csv('Metrics\\Decision_Tree_Metrics.csv', index=False)
    with open('Metrics\\Decision_Tree_Classification_Report.csv', 'w') as f:
        f.write(classification_rep_DT)

# 3.2.2. Store Random Forest Metrics
def classification_performance_store_rf(
        accuracy_RF, 
        f1_RF, 
        classification_rep_RF
    ):
    print("Classification # 3.2.2. Store Random Forest Metrics (In ./Metrics)")
    metrics = {
        'Metric': ['Accuracy', 'F1-score'],
        'Value': [accuracy_RF, f1_RF]}
    df = pd.DataFrame(metrics)
    df.to_csv('Metrics\\Random_Forest_Metrics.csv', index=False)
    with open('Metrics\\Random_Forest_Classification_Report.csv', 'w') as f:
        f.write(classification_rep_RF)

# 3.2.3. Store KNN Metrics
def classification_performance_store_knn(
        accuracy_KNN, 
        f1_KNN, 
        classification_rep_KNN
    ):
    print("Classification # 3.2.3. Store KNN Metrics (In ./Metrics)")
    metrics = {
        'Metric': ['Accuracy', 'F1-score'],
        'Value': [accuracy_KNN, f1_KNN]}
    df = pd.DataFrame(metrics)
    df.to_csv('Metrics\\KNN_Metrics.csv', index=False)
    with open('Metrics\\KNN_Classification_Report.csv', 'w') as f:
        f.write(classification_rep_KNN)

# 3.2.4. Store GBM Metrics
def classification_performance_store_gbm(
        accuracy_GBM, 
        f1_GBM, 
        classification_rep_GBM
    ):
    print("Classification # 3.2.4. Store GBM Metrics (In ./Metrics)")
    metrics = {
        'Metric': ['Accuracy', 'F1-score'],
        'Value': [accuracy_GBM, f1_GBM]
    }
    df = pd.DataFrame(metrics)
    df.to_csv('Metrics\\GBM_Metrics.csv', index=False)
    with open('Metrics\\GBM_Classification_Report.csv', 'w') as f:
        f.write(classification_rep_GBM)

# 4. Feature Importance Analysis
def classification_importance(
        dt_classifier, rf_classifier, knn_classifier, gbm_classifier,
        X_DT, X_RF, X_GBM, 
        X_KNN_test, y_KNN_test, 
        knn_predictions
    ):
    print("Classification # 4. Feature Importance")
    DT_Importance, RF_Importance, GBM_Importance, importances_sorted_KNN = classification_importance_calculate(
        dt_classifier, rf_classifier, knn_classifier, gbm_classifier,
        X_DT, X_RF, X_GBM, 
        X_KNN_test, y_KNN_test, 
        knn_predictions
    ) # 4.1.
    classification_importance_result(
        *DT_Importance, 
        *RF_Importance,
        *GBM_Importance,
        importances_sorted_KNN
    ) # 4.2.

# 4.1. Calculation of Importance
def classification_importance_calculate(
        dt_classifier, rf_classifier, knn_classifier, gbm_classifier,
        X_DT, X_RF, X_GBM, 
        X_KNN_test, y_KNN_test, 
        knn_predictions
    ):
    print("Classification # 4.1. Calculation of Importance")
    DT_Importance, RF_Importance, GBM_Importance = classification_importance_calculate_loss(
        dt_classifier, rf_classifier, gbm_classifier, 
        X_DT, X_RF, X_GBM
    ) # 4.1.1.
    importances_sorted_KNN = classification_importance_calculate_permutation(
        knn_classifier, 
        X_KNN_test, 
        y_KNN_test, 
        knn_predictions
    ) # 4.1.2.
    return DT_Importance, RF_Importance, GBM_Importance, importances_sorted_KNN

# 4.1.1. Importance for GBM/TD/RF (Loss reduction method)
def classification_importance_calculate_loss(
        dt_classifier, rf_classifier, gbm_classifier, 
        X_DT, X_RF, X_GBM
    ):
    print("Classification # 4.1.1. Importance for GBM/TD/RF (Loss reduction method)")
    feature_importances_DT, feature_names_DT, sorted_idx_DT = classification_importance_calculate_loss_dt(
        dt_classifier, 
        X_DT
    ) # 4.1.1.1.
    feature_importances_RF, feature_names_RF, sorted_idx_RF = classification_importance_calculate_loss_rf(
        rf_classifier, 
        X_RF
    ) # 4.1.1.2.
    feature_importances_RF, feature_names_RF, sorted_idx_RF = classification_importance_calculate_loss_gbm(
        gbm_classifier, 
        X_GBM
    ) # 4.1.1.3.
    DT_Importance = (
        feature_importances_DT, 
        feature_names_DT, 
        sorted_idx_DT
    )
    RF_Importance = (
        feature_importances_RF, 
        feature_names_RF, 
        sorted_idx_RF
    )
    GBM_Importance = (
        feature_importances_RF, 
        feature_names_RF, 
        sorted_idx_RF
    )
    return DT_Importance, RF_Importance, GBM_Importance

# 4.1.1.1. Calculate importance for Decision Tree
def classification_importance_calculate_loss_dt(
        dt_classifier, 
        X_DT
    ):
    print("Classification # 4.1.1.1. Importance for Decision Tree")
    feature_importances_DT = dt_classifier.feature_importances_
    feature_names_DT = X_DT.columns
    sorted_idx_DT = feature_importances_DT.argsort()
    return feature_importances_DT, feature_names_DT, sorted_idx_DT

# 4.1.1.2. Calculate importance for Random Forest
def classification_importance_calculate_loss_rf(
        rf_classifier, 
        X_RF
    ):
    print("Classification # 4.1.1.2. Importance for Random Forest")
    feature_importances_RF = rf_classifier.feature_importances_
    feature_names_RF = X_RF.columns
    sorted_idx_RF = feature_importances_RF.argsort()
    return feature_importances_RF, feature_names_RF, sorted_idx_RF

# 4.1.1.3. Calculate importance for GBM
def classification_importance_calculate_loss_gbm(
        gbm_classifier, 
        X_GBM
    ):
    print("Classification # 4.1.1.3. Importance for GBM")
    feature_importances_gbm = gbm_classifier.feature_importances_
    feature_names_GBM = X_GBM.columns
    sorted_idx_gbm = feature_importances_gbm.argsort()
    return feature_importances_gbm, feature_names_GBM, sorted_idx_gbm

# 4.1.2. Calculate permutation importance for KNN
def classification_importance_calculate_permutation(
        knn_classifier, 
        X_KNN_test, 
        y_KNN_test, 
        knn_predictions
    ):
    print("Classification # 4.1.2. Importance for KNN (Permutation)")
    original_accuracy_KNN = accuracy_score(y_KNN_test, knn_predictions)
    importances_KNN = {}
    for feature in X_KNN_test.columns:
        X_test_shuffled_KNN = X_KNN_test.copy()
        X_test_shuffled_KNN[feature] = np.random.permutation(X_test_shuffled_KNN[feature].values)
        shuffled_predictions_KNN = knn_classifier.predict(X_test_shuffled_KNN)
        shuffled_accuracy_KNN = accuracy_score(
            y_KNN_test, 
            shuffled_predictions_KNN
        )
        importances_KNN[feature] = max(0, original_accuracy_KNN - shuffled_accuracy_KNN)
    importances_sorted_KNN = sorted(importances_KNN.items(), key=lambda x: x[1], reverse=True)
    return importances_sorted_KNN


# 4.2. Display feature importance results
def classification_importance_result(
        feature_importances_DT, feature_names_DT, sorted_idx_DT,                 
        feature_importances_RF, feature_names_RF, sorted_idx_RF,
        feature_importances_gbm, feature_names_GBM, sorted_idx_gbm,
        importances_sorted_KNN
    ):
    print("Classification # 4.2. Importance Result (In ./Plots)")
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    axes[0, 0].barh(
        range(len(sorted_idx_DT)), 
        feature_importances_DT[sorted_idx_DT], 
        align='center', 
        color="blue"
    )
    axes[0, 0].set_yticks(range(len(sorted_idx_DT)))
    axes[0, 0].set_yticklabels([feature_names_DT[i] for i in sorted_idx_DT])
    axes[0, 0].set_xlabel('Feature Importance')
    axes[0, 0].set_title('Feature Importance (Decision Tree)')
    axes[0, 1].barh(
        range(len(sorted_idx_RF)), 
        feature_importances_RF[sorted_idx_RF], 
        align='center', 
        color="blue"
    )
    axes[0, 1].set_yticks(range(len(sorted_idx_RF)))
    axes[0, 1].set_yticklabels([feature_names_RF[i] for i in sorted_idx_RF])
    axes[0, 1].set_xlabel('Feature Importance')
    axes[0, 1].set_title('Feature Importance (Random Forest)')
    axes[1, 0].barh(
        range(len(sorted_idx_gbm)), 
        feature_importances_gbm[sorted_idx_gbm],
        align='center', 
        color="blue"
    )
    axes[1, 0].set_yticks(range(len(sorted_idx_gbm)))
    axes[1, 0].set_yticklabels([feature_names_GBM[i] for i in sorted_idx_gbm])
    axes[1, 0].set_xlabel('Feature Importance')
    axes[1, 0].set_title('Feature Importance (GBM)')
    axes[1, 1].barh(
        [imp[0] for imp in importances_sorted_KNN][::-1], 
        [imp[1] for imp in importances_sorted_KNN][::-1], 
        color="blue"
    )
    axes[1, 1].set_xlabel('Importance (Drop in Accuracy)')
    axes[1, 1].set_title('Feature Importance by Accuracy Degradation (KNN)')
    plt.tight_layout()
    plt.savefig('Plots\\Importance.png')
    plt.close()