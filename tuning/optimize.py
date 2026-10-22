import yaml
import optuna
import tensorflow as tf

from optuna.integration import TFKerasPruningCallback

from training.train import prepare_training_data_split
from models.gru import GRUModel

# ============================================================
# CONFIG
# ============================================================

with open("config/dataset.yaml") as file:
    dataset_config = yaml.safe_load(file)

dataset = dataset_config["dataset"]

def objective(trial):

    # ========================================================
    # HYPERPARAMETERS
    # ========================================================

    n = trial.suggest_categorical(
	"n",
	[3, 5, 7, 9]
    )
    embedding_dim = trial.suggest_categorical(
	"embedding_dim",
	[16, 32, 64, 128, 256]
    )
    gru_units_1 = trial.suggest_categorical(
	"gru_units",
	[32, 64, 128, 256]
    )
    gru_units_2 = trial.suggest_categorical(
        "gru_units_2",
        [32, 64, 128, 256]
    )
    dense_units = trial.suggest_categorical(
        "dense_units",
        [32, 64, 128, 256]
    )
    dropout = trial.suggest_float(
	"dropout",
	0.0,
	0.5,
	step=0.1
    )
    batch_size = trial.suggest_categorical(
	"batch_size",
	[512, 1024, 2048]
    )
    learning_rate = trial.suggest_float(
	"learning_rate",
	1e-5,
	1e-2,
	log=True
    )

    # ========================================================
    # DATA
    # ========================================================

    X_train, X_val, y_train, y_val, id_to_index, index_to_id = prepare_training_data_split(
        dataset["path"],
        dataset["train"]["directory"],
        dataset["train"]["files"],
        n,
        validation_ratio=0.2
    )

    num_ids = len(id_to_index)

    # ========================================================
    # MODEL
    # ========================================================

    model = GRUModel(
        num_ids=num_ids,
        embedding_dim=embedding_dim,
        gru_units_1=gru_units_1,
        gru_units_2=gru_units_2,
        dense_units=dense_units,
        dropout=dropout
    )

    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
	optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
	),
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"]
    )

    # ========================================================
    # CALLBACKS
    # ========================================================
    callbacks = [
        TFKerasPruningCallback(
            trial,
            monitor="val_loss"
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    history = model.fit(
 	X_train,
	y_train,
	validation_data=(X_val, y_val),
	batch_size=batch_size,
	epochs=50,
	callbacks=callbacks,
	verbose=0
    )

    # METRIC FOR OPTUNA
    return min(history.history["val_loss"])

# ============================================================
# OPTUNA STUDY
# ============================================================

study = optuna.create_study(
    study_name="gru_stacked_optimization",
    storage="sqlite:///gru_stacked_optuna.db",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
    ),
)

study.optimize(
    objective,
    n_trials=10,
)

# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("OPTIMIZATION FINISHED")
print("========================================")

print("Best validation loss:")
print(study.best_value)

print("\nBest hyperparameters:")

for parameter, value in study.best_params.items():
    print(f"{parameter}: {value}")

# ============================================================
# SAVE BEST CONFIGURATION
# ============================================================

best_params = study.best_params

model_tuned_config = {
    "sliding_window": {
        "n": best_params["n"]
    },

    "model": {
        "embedding_dim": best_params["embedding_dim"],
        "gru_units_1": best_params["gru_units"],
        "gru_units_2": best_params["gru_units_2"],
        "dense_units": best_params["dense_units"],
        "dropout": best_params["dropout"]
    },

    "training": {
        "batch_size": best_params["batch_size"],
        "epochs": 50,
        "learning_rate": best_params["learning_rate"]
    }
}

with open("config/model_tuned.yaml", "w") as file:
    yaml.safe_dump(
        model_tuned_config,
        file,
        sort_keys=False
    )

print("\nBest configuration saved to:")
print("config/model_tuned.yaml")
