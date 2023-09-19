import os
import re
import warnings
from typing import List

import joblib
import numpy as np
import pandas as pd
from pycaret.regression import compare_models, finalize_model, predict_model, setup
from sklearn.exceptions import NotFittedError

from schema.data_schema import RegressionSchema

warnings.filterwarnings("ignore")

os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

PREDICTOR_FILE_NAME = "predictor.joblib"


def clean_and_ensure_unique(names: List[str]) -> List[str]:
    """
    Clean the provided column names by removing special characters and ensure their
    uniqueness.

    The function first removes any non-alphanumeric character (except underscores)
    from the names. Then, it ensures the uniqueness of the cleaned names by appending
    a counter to any duplicates.

    Args:
        names (List[str]): A list of column names to be cleaned.

    Returns:
        List[str]: A list of cleaned column names with ensured uniqueness.

    Example:
        >>> clean_and_ensure_unique(['3P%', '3P', 'Name', 'Age%', 'Age'])
        ['3P', '3P_1', 'Name', 'Age', 'Age_1']
    """

    # First, clean the names
    cleaned_names = [re.sub("[^A-Za-z0-9_]+", "", name) for name in names]

    # Now ensure uniqueness
    seen = {}
    for i, name in enumerate(cleaned_names):
        original_name = name
        counter = 1
        while name in seen:
            name = original_name + "_" + str(counter)
            counter += 1
        seen[name] = True
        cleaned_names[i] = name

    return cleaned_names


class Regressor:
    """A wrapper class for the PyCaret Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    def __init__(self, train_input: pd.DataFrame, schema: RegressionSchema):
        """Construct a new PyCaret Regressor.

        Args:
           train_input (pd.DataFrame): The input data for model training.
           schema (RegressionSchema): Schema of the input data.
        """
        self._is_trained = False
        self.train_input = train_input
        self.schema = schema
        self.setup(train_input, schema)
        self.model = self.compare_models()
        self.model_name = "pycaret_regressor_model"

    def setup(self, train_input: pd.DataFrame, schema: RegressionSchema):
        """Set up the experiment of comparing different models.
        Args:
            train_input: The data  of training including the target column.
            schema: schema of the provided data.
        """
        setup(
            train_input,
            target=schema.target,
            remove_outliers=True,
            normalize=True,
            ignore_features=[schema.id],
        )

    def compare_models(self):
        """Compares multiple regression models."""
        self._is_trained = True
        return compare_models()

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.model.predict(inputs)

    def save(self, model_dir_path: str) -> None:
        """Save the regressor model to disk.
        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        pipeline = finalize_model(self.model)
        joblib.dump(pipeline, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def predict_with_model(regressor: "Regressor", data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict class probabilities for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted label.
    """
    return predict_model(regressor, data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)
