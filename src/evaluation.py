from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def evaluate_model(
    y_true,
    y_pred,
    y_probs
):

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_true,
            y_probs
        )
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )