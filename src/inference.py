import joblib


def load_model(path):

    return joblib.load(path)


def predict_transaction(
    model,
    sample
):

    prediction = model.predict(
        sample
    )[0]

    probability = model.predict_proba(
        sample
    )[:,1][0]

    return {

        "prediction":
        int(prediction),

        "fraud_probability":
        float(probability)
    }