from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def upsampling(ratio,majority,minority):
    df_minority_upsampled = resample(minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=int(round(len(majority)*(ratio/(1-ratio)))),    # to match majority class
                                 random_state=123) # reproducible results
 
    ## Combine majority class with upsampled minority class
    df_upsampled = pd.concat([majority, df_minority_upsampled])
 
    # Display new class counts
    print(df_upsampled.is_attributed.value_counts(normalize=True))
    return df_upsampled

def downsampling(ratio,majority,minority):
    df_majority_downsampled = resample(majority, 
                                 replace=True,     # sample with replacement
                                 n_samples=int(round(len(minority)*((1-ratio)/ratio))),   # to match majority class
                                 random_state=123) # reproducible results
 
    ## Combine majority class with upsampled minority class
    df_downsampled = pd.concat([df_majority_downsampled, minority])
 
    # Display new class counts
    print(df_downsampled.target.value_counts(normalize=True))
    return df_downsampled

def smote_transform(ratio):
    sm = SMOTE(random_state=42,sampling_strategy=ratio/(1-ratio))
    X_smote, y_smote = sm.fit_resample(X_train,y_train)
    print(y_smote.value_counts(normalize=True))
    return X_smote,y_smote


# modeling function
def prediction_model(data_X,data_Y,name,clf):

    classifier=clf
    clf.fit(data_X, data_Y)
    predicted = clf.predict(X_test)  # prediction for validation 
    fitted_y=clf.predict(data_X)   # prediction for train 
    y_prob=clf.predict_proba(X_test)[:,1] # predict proba for validation 
    y_prob_train=clf.predict_proba(data_X)[:,1] 
    print("Processing...")
    
    confusion_matrix =  pd.crosstab(index=y_test, 
                                    columns=predicted.ravel(), 
                                    rownames=['Expected'], 
                                    colnames=['Predicted'])
    
    accuracy = np.round(accuracy_score(y_test , predicted),3)
    sns.heatmap(confusion_matrix, annot=True, square=False, fmt='', cbar=False)
    plt.title(name + ", Test Accuracy: " + str(accuracy), fontsize = 15)
    plt.show()
    
    brier=brier_score_loss(y_test, y_prob)

  
    brier_train=brier_score_loss(data_Y, y_prob_train)

    train_confusion_matrix =  pd.crosstab(index=data_Y, 
                                    columns=fitted_y.ravel(), 
                                    rownames=['Expected'], 
                                    colnames=['Predicted'])
    
    accuracy_train = np.round(accuracy_score(data_Y , fitted_y),3)
    sns.heatmap(train_confusion_matrix, annot=True, square=False, fmt='', cbar=False)
    plt.title(name + ", Train Accuracy: " + str(accuracy_train), fontsize = 15)
    plt.show()
    
    print("Model parameters ", "\n", classifier)
    
    print("Classification report for classifier in test ","\n",
               classification_report(y_test, predicted))
    print("Classification report for classifier in train ","\n",
              classification_report(data_Y, fitted_y))
    
    print("Brier Score for Train: ","\n",brier_train)
    print("Brier Score for Test: ","\n",brier)

# get performance for selected ratios 
ratios=list(np.arange(0.05,0.55,0.05))
# below is sudo code for how to get performance for selected ratios
for ratio in ratios:
    names= str(round(ratio,3))+' Sampling techiniques we select + Baseline Model'
    X,y = array after using sampling techiniques 
    prediction_model(X,y,name=names,clf)
    print('\n')

