
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,f1_score
import sklearn
import joblib
import boto3
import joblib
import pathlib
from io import StringIO
import argparse
import numpy as np
import pandas as pd
import os


def model_fn(model_dir):
    clf =joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__== "__main__":
    
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()
    #hyperparameters sent by the client are passed as command-line arguments to the parser.add_argument("--n_estimators", type=int, default=100) parser.add_argument("--random_state", type=int, default=0)
    # Data, model, and output directories
    
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=90)
    
    
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")
    
    args,_ = parser.parse_known_args()
    
    print("SKLEARN VERSION: ",sklearn.__version__)
    print("JOBLIB VERSION: ",joblib.__version__)
    
    print('[INFO] Reading Data')
    print()
    
    train_df = pd.read_csv(os.path.join(args.train,args.train_file))
    test_df = pd.read_csv(os.path.join(args.test,args.test_file))
    
    features = list(train_df.columns)
    label = features.pop(-1)
    
    print('Building Training and Testing Datasets')
    print()
    
    X_train = train_df[features]
    X_test = test_df[features]
    y_train= train_df[label]
    y_test = test_df[label]
    
    print('Column order:')
    print(features)
    print()
    
    print('Label column:',label)
    print()
    
    
    print('Data Shape: ')
    print()
    
    print('---SHAPE OF TRAINING DATA ---')
    print(X_train.shape)
    print(X_test.shape)
    print()
    
    
    print('---SHAPE OF TEST DATA ---')
    print(X_test.shape)
    print(y_test.shape)
    print()
    
    
    print("Training Random Forest Model......")
    print()
    model = RandomForestClassifier(n_estimators=args.n_estimators,random_state=args.random_state,verbose=2)
    model.fit(X_train,y_train)
    print()
    
    
    model_path = os.path.join(args.model_dir,"model.joblib")
    joblib.dump(model,model_path)
    print('Model Persisted at'+ model_path)
    print()
    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred_test)
    test_rep=classification_report(y_test,y_pred_test)
    
    
    
    print()
    print("----METRICS RESULTS FOR TESTING DATA------")
    print()
    print("TOTAL ROWS ARE:", X_test.shape[0])
    print("[TESTING] ACCURACY IS :",test_acc)
    print('[TESTING] Testing Report')
    print(test_rep)
    
    
   
