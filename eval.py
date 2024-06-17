from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import roc_auc_score




def cv_auroc(df, target, model, cv):

    # container for storing AUROC of each fold
    aurocs = []

    for fold, (train, test) in enumerate(cv.split(df, target)):
        # for each fold, fit the model and use it to predict class probabilities of test data
        model.fit(df.iloc[train], target.iloc[train])
        y_pred = model.predict_proba(df.iloc[test])[:, 1]

        # then calculate the auroc by comparing model predictions to test labels
        score = roc_auc_score(target.iloc[test], y_pred)

        # add the auroc score for the fold to the array
        aurocs.append(score)

    # take the mean and std dev of the auroc scores and return them
    return np.mean(aurocs), np.std(aurocs)





def top_n_eval(dfs, target):

    # extract integer keys of the dfs dictionary, which corresponding how many top features were selected
    keys = dfs.keys()

    for i in keys:
        # announce how many top genes we are working with
        print("Analyzing model performance with the top", i, "genes:")

        # for each key, attain the corresponding dataframe from the input dictionary
        full_df = dfs[i]

        # set up cross validation mechanisms: a ShuffleSplit object will result in the same stratification
        # if used across multiple models. So, we must instantiate multiple objects
        cv1 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=3)
        cv2 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=3)
        cv3 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=3)

        # first, use a cross-validated RF classifier. Start by instantiating model, then run cross_val_score for
        # different scoring metrics.
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=3)
        # accuracy
        rf_acc = cross_val_score(rf, full_df, target, cv=cv1, scoring='accuracy')
        print("The cross-validated mean accuracy of the RF model is", rf_acc.mean(), "w/ std", rf_acc.std())
        rf_f1 = cross_val_score(rf, full_df, target, cv=cv1, scoring='f1')
        print("The cross-validated mean f1 score of the RF model is", rf_f1.mean(), "w/ std", rf_f1.std())
        rf_mean_auroc, rf_std_auroc = cv_auroc(full_df, target, rf, cv1)
        print("The cross-validated mean AUROC of the RF model is", rf_mean_auroc, "w/ std", rf_std_auroc)

        # next, use a cross-validated SVM classifier. The linear kernel (k(x,z) = x^T*z) is chosen since the number
        # of features dwarfs the number of samples in the dataset, so increasing dimensions will not improve
        # seperability.
        svm = SVC(kernel='linear', probability=True, random_state=3)
        svm_acc = cross_val_score(svm, full_df, target, cv=cv2, scoring='accuracy')
        print("The cross-validated mean accuracy of the SVM model is", svm_acc.mean(), "w/ std", svm_acc.std())
        svm_f1 = cross_val_score(svm, full_df, target, cv=cv2, scoring='f1')
        print("The cross-validated mean f1 score of the SVM model is", svm_f1.mean(), "w/ std", svm_f1.std())
        svm_mean_auroc, svm_std_auroc = cv_auroc(full_df, target, svm, cv2)
        print("The cross-validated mean AUROC of the SVM model is", svm_mean_auroc, "w/ std", svm_std_auroc)

        # now use a cross-validated KNN classifier. Number of neighbors arbitrarily chosen to be sqrt(n)
        n = len(full_df)
        k = int(np.sqrt(n))
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_acc = cross_val_score(knn, full_df, target, cv=cv3, scoring='accuracy')
        print("The cross-validated mean accuracy of the KNN model is", knn_acc.mean(), "w/ std", knn_acc.std())
        knn_f1 = cross_val_score(knn, full_df, target, cv=cv3, scoring='f1')
        print("The cross-validated mean f1 score of the KNN model is", knn_f1.mean(), "w/ std", knn_f1.std())
        knn_mean_auroc, knn_std_auroc = cv_auroc(full_df, target, knn, cv3)
        print("The cross-validated mean AUROC of the KNN model is", knn_mean_auroc, "w/ std", knn_std_auroc)

    return
