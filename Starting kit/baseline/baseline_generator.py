from sklearn.dummy import DummyClassifier
import pandas as pd
import pickle as pkl
import csv
import os

# Reads train data and returns two lists of texts and labels
def read_train_data(fpath, delim='\t'):
    data = pd.read_csv(fpath, delimiter=delim)
    X = list(data['Text'])
    y = list(data['Labels'])
    return X, y

# Reads test data and returns two lists of IDs and labels
def read_test_data(fpath, delim='\t'):
    data = pd.read_csv(fpath, delimiter=delim)
    ids = list(data['ID'])
    data = list(data['Text'])
    return ids, data

# Trains and generats a dummy classifier with the given strategy
def generate_dummy_classifier(X, y, strategy):
    dummy_clf = DummyClassifier(strategy=strategy, random_state=seed)
    dummy_clf.fit(X, y)
    return dummy_clf

# Pickles the classifier
def save_classifier(clf, path):
    with open (path, 'wb') as handler:
        pkl.dump(clf, handler)

# Predicts the classes
def predict_classes(clf, y_test, fname):
    y_pred = clf.predict(y_test)
    write_preds(y_test, y_pred, fname)

# Write the predictions to the given file
def write_preds(input, preds, fname, delim='\t'):
    with open (fname, 'w') as f_w:
        writer = csv.writer(f_w, delimiter=delim)
        writer.writerow(['ID', 'Labels'])
        for id, label in zip(input, preds):
            writer.writerow([id, label])



##### ================= Main ============================ #####
if __name__=="__main__":
    #####====== Editable variables (change as per your requirement) ======####
    seed = 100 # random seed to be used for some strategies
    root = '../training_data' #path to the training data directory
    model_path = 'models' #path where the model files will be pickled
    predictions_path = 'predictions' #path where the predictions will be saved
    langs = ['hin', 'ben', 'mni', 'multi']  # languages for which classifiers are to be trained
    strategies = ['uniform', 'stratified', 'most_frequent', 'prior'] #strategies to use for generating dummy classifier
    isSave = True # If true then the model file is pickled
    isPredict = True # If true then predictions are generated and saved (need to provide path to the test file)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    for lang in langs:
        print ('Language', lang)
        
        trpath=os.path.join(root, 'train_'+lang+'.tsv') # Path to the train file
        tspath=os.path.join(root, 'dev_'+lang+'.tsv') # Path to the test file
        
        X, y = read_train_data(trpath) #train data
        test_ids, test_data = read_test_data(tspath) #test data

        for strategy in strategies:
            print ('Strategy', strategy)
            clf = generate_dummy_classifier(X, y, strategy)
            if isSave:
                print ('Pickling classifier')
                fpath = os.path.join(model_path, strategy +'_' + lang + '.pkl')
                save_classifier(clf, fpath)
            if isPredict:
                print ('Predicting and writing classes')
                wpath = os.path.join(predictions_path, strategy)
                if not os.path.exists(wpath):
                    os.makedirs(wpath)
                fname = os.path.join(wpath, 'pred_'+lang+'.tsv')
                predict_classes(clf, test_ids, fname)


