def mean_absolute_error(y_true, y_pred):
    """
    Calculate the MAE between two df
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lists don't have the same length")

    absolute_errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]

    mae = sum(absolute_errors) / len(y_true)

    round_mae = round(mae, 2)
    return round_mae
