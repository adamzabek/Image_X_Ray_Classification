import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from cnn_model import build_cnn, train_cnn


def cnn_mlflow_run(name, cnn_params, train_params, train_dataset, y_train, val_dataset, y_val, test_dataset, y_test):
    
    """
    
        INPUT:
        name - name of experiment in mlflow
        cnn_params - paramaters to build cnn architecture
        train_params - parameters to train cnn model
        train_dataset - train dataset
        y_train - y labels for train dataset
        val_dataset - validation dataset
        y_val - y labels for validation dataset
        test_dataset - test dataset
        y_test - y labels for test dataset
        
        OUTPUT:
        trained model saved in remote mlflow server
        
    """    
    
    with mlflow.start_run(run_name=name):
        
        mlflow.log_params(cnn_params)
        
        mlflow.log_params(train_params)
        
        mlflow.set_tag("cnn_model", "cnn")
        
        cnn = build_cnn(cnn_params)
        
        cnn = train_cnn(cnn, train_params, train_dataset, val_dataset, y_train, y_val)
        
        x_test_tensor = tf.cast(np.array(test_dataset)/255, dtype=tf.float32)

        test_preds = (cnn.predict(x_test_tensor) > 0.5).astype("int32")
        
        test_preds = test_preds.reshape(1,-1)[0]
        
        test_acc = accuracy_score(y_test, test_preds)
        
        mlflow.log_metric("test_acc", test_acc)
        
        mlflow.tensorflow.log_model(cnn, "cnn_model")