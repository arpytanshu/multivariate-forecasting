import numpy as np
from typing import Dict, List

def perform_covariate_op(encoded_date: Dict, type: str, func: str) -> np.ndarray:
    """A helper function to perform covariates op

    Parameters
    ----------
    values : np.ndarray
        the values to perform op on 
    type : str
        type of value : ['hour','weekday']
    func : str
        function to use : ['sin', 'cos']

    Returns
    -------
    np.ndarray
        Output of operation
    """

    functions = {"sin": np.sin, "cos": np.cos}
    types = {"hour": 24, "weekday": 7, "minute": 60}
    result = functions[func]((encoded_date[type] * 2 * np.pi) / types[type])
    return result


def generate_time_covariates(unix_timestamps: np.int64, covariates_switch:Dict)->List[np.ndarray]:
    """A helper function to generate time based covariates using timestamps

    Parameters
    ----------
    unix_timestamps : np.int64
        list of unix timestamps of the ts
    covariates_switch : Dict
        Config dict to toggle certain covariates

    Returns
    -------
    List[np.ndarray]
        list of computed covariates
    """

    encoded_date = {
        "weekday": (unix_timestamps // 86400 ) % 7,
        "hour": ( unix_timestamps // 3600 ) % 24,
        "minute": ( unix_timestamps // 60 ) % 60,
    }
    
    covariates = [
        perform_covariate_op(encoded_date, *val_type.split("_"))
        for val_type, enabled in covariates_switch.items()
        if enabled
    ]

    return covariates

