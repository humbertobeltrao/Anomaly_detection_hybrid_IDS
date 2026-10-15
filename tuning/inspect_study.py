import optuna


STUDY_NAME = "gru_hyperparameter_optimization"
STORAGE = "sqlite:///gru_optuna.db"


def main():

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE
    )

    print("========================================")
    print("OPTUNA STUDY")
    print("========================================")

    print("Study:", study.study_name)
    print("Trials:", len(study.trials))

    print("\n========================================")
    print("TRIALS")
    print("========================================")

    for trial in study.trials:

        print(
            f"\nTrial {trial.number}"
        )

        print("  State:", trial.state.name)
        print("  Value:", trial.value)

        print("  Parameters:")

        for parameter, value in trial.params.items():
            print(f"    {parameter}: {value}")

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    pruned_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.PRUNED
    ]

    failed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.FAIL
    ]

    print("\n========================================")
    print("SUMMARY")
    print("========================================")

    print("Completed:", len(completed_trials))
    print("Pruned:", len(pruned_trials))
    print("Failed:", len(failed_trials))

    if completed_trials:

        print("\n========================================")
        print("BEST TRIAL")
        print("========================================")

        best_trial = study.best_trial

        print("Trial:", best_trial.number)
        print("Validation loss:", best_trial.value)

        print("\nBest hyperparameters:")

        for parameter, value in best_trial.params.items():
            print(f"  {parameter}: {value}")


if __name__ == "__main__":
    main()
