import pandas as pd
import numpy as np


def create_log_feature(df):

    df["Amount_log"] = np.log1p(
        df["Amount"]
    )

    return df