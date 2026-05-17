from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def build_model():

    pipeline = Pipeline([

        (
            "model",

            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
        )

    ])

    return pipeline