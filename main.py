import os
import mlflow
import numpy as np
import tensorflow as tf
from train_model import cnn_mlflow_run
from train_data_loader import train_data_loader
from sklearn.model_selection import train_test_split



remote_server_uri = "*****" ### insert url to remote server
mlflow.set_tracking_uri(remote_server_uri)

os.environ['MLFLOW_TRACKING_USERNAME'] = '*****' ### insert name
os.environ['MLFLOW_TRACKING_PASSWORD'] = '*****' ### insert password

mlflow.set_experiment("Image_Classification")


def main():
    
    img_size = 256
    
    data_dir = './dataset/raw_data'
    
    labels = ['NORMAL', 'PNEUMONIA']
    
    print("Start preprocessing data...")
    
    preprocessed_data, y_data = train_data_loader(data_dir, labels, img_size)
    
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        preprocessed_data,
        y_data,
        test_size = 0.15, 
        random_state = 0,
        stratify = y_data)
    
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, 
        y_val_test,
        test_size = 0.20, 
        random_state = 0,
        stratify = y_val_test)
    
    x_train = np.array(x_train).reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    
    x_val = np.array(x_val).reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val).reshape(-1, 1)
    
    x_train_tensor = tf.cast(x_train, dtype=tf.float32)
    x_val_tensor = tf.cast(x_val, dtype=tf.float32)
    
    print("Finish preprocessing data...")
    
    cnn_params = {
        "conv_layer_size_1": 32,
        "conv_layer_size_2": 64,
        "conv_layer_size_3": 128,
        "dense_layer_1": 120, 
        "dense_layer_2": 60,
        "dropout_rate": 0.2,
        "pool_size_1": 2,
        "pool_size_2": 3,
        "kernel": 3,
        "dense_size": 1,
        "input_size": 256,
        "channel": 1,
        "activation_conv": 'relu',
        "activation_dense": 'sigmoid',
        "padding": 'same'
    }

    train_params = {
        "patience": 6,
        "epochs": 200
    }

    print("Start training model...")

    cnn_mlflow_run(
        "cnn_image_classification",
        cnn_params,
        train_params,
        x_train_tensor,
        y_train,
        x_val_tensor,
        y_val,
        x_test,
        y_test
    )
    
    print("Finish training model...")

if __name__ == "__main__":
    main()