import os
import tensorflow as tf
import pandas as pd

from preprocessing.load_data import (get_file_paths, load_multiple_ids, create_id_mapping, encode_ids)
from preprocessing.sliding_window import create_sliding_windows
from preprocessing.split_data import train_validation_split

from models.gru import GRUModel

def prepare_training_data(road_path, directory, filenames, n):
    """
    Load and encode the training CAN ID sequence.
    """

    file_paths = get_file_paths(
        road_path,
        directory,
        filenames
    )

    ids = load_multiple_ids(file_paths)

    id_to_index, index_to_id = create_id_mapping(ids)

    encoded_ids = encode_ids(
        ids,
        id_to_index
    )

    X, y = create_sliding_windows(
        encoded_ids,
        n
    )

    return X, y, id_to_index, index_to_id

def prepare_training_data_split(road_path, directory, filenames, n, validation_ratio=0.2):
    """
    Prepare the training and validation data from benign CAN IDs.
    """

    X, y, id_to_index, index_to_id = prepare_training_data(
        road_path,
        directory,
        filenames,
        n
    )

    X_train, X_val, y_train, y_val = train_validation_split(
        X,
        y,
        validation_ratio
    )

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        id_to_index,
        index_to_id
    )

def create_model(config, num_ids):
    """
    Create the stacked GRU model from the configuration.
    """

    model = GRUModel(
        num_ids=num_ids,
        embedding_dim=config["model"]["embedding_dim"],
        gru_units_1=config["model"]["gru_units_1"],
        gru_units_2=config["model"]["gru_units_2"],
        dense_units=config["model"]["dense_units"],
        dropout=config["model"]["dropout"]
    )

    return model

def compile_model(model, config):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

def train_model(model, X_train, y_train, X_val, y_val, config):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
	verbose=1
    )

    return history

def save_training_history(history, output_path):
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )
    history_data = history.history
    history_df = pd.DataFrame(history_data)

    history_df.index += 1
    history_df.index.name = "epoch"

    history_df.to_csv(output_path)

def run_training(dataset_config, model_config, results_dir):
    """
    Execute the complete training pipeline.
    """

    dataset = dataset_config["dataset"]

    # 1. Prepare training data
    X_train, X_val, y_train, y_val, id_to_index, index_to_id = (
        prepare_training_data_split(
            dataset["path"],
            dataset["train"]["directory"],
            dataset["train"]["files"],
            model_config["sliding_window"]["n"],
            validation_ratio=0.2
        )
    )

    # 2. Create model
    num_ids = len(id_to_index)

    model = create_model(
        model_config,
        num_ids
    )

    # 3. Compile model
    compile_model(
        model,
        model_config
    )

    # 4. Train model
    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
	y_val,
        model_config
    )

    save_training_history(
        history,
        os.path.join(results_dir, "training_history.csv")
    )

    return model, history, id_to_index, index_to_id, X_val, y_val
