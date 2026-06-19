import optuna
from optuna_objective import objective


def main():

    study = optuna.create_study(
        study_name="yolo_search",
        storage="sqlite:///optuna.db",
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    main()