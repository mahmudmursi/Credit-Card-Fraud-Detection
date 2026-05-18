import numpy as np


def predict_transaction(model, sample):

    # Create Amount_log feature
    sample["Amount_log"] = np.log1p(sample["Amount"])

    # Match exact training features
    sample = sample[model.feature_names_in_]

    prediction = int(model.predict(sample)[0])

    probability = float(
        model.predict_proba(sample)[0][1]
    )

    return {
        "prediction": prediction,
        "fraud_probability": probability
    }