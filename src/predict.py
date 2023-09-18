from config import paths
from utils import read_csv_in_directory, save_dataframe_as_csv
from logger import get_logger
from Regressor import Regressor, predict_with_model
from schema.data_schema import load_saved_schema

logger = get_logger(task_name="predict")


def run_batch_predictions() -> None:
    """
        Run batch predictions on test data, save the predicted probabilities to a CSV file.

        This function reads test data from the specified directory,
        loads the preprocessing pipeline and pre-trained predictor model,
        transforms the test data using the pipeline,
        makes predictions using the trained predictor model,
        adds ids into the predictions dataframe,
        and saves the predictions as a CSV file.
        """
    x_test = read_csv_in_directory(paths.TEST_DIR)
    data_schema = load_saved_schema(paths.SAVED_SCHEMA_DIR_PATH)
    model = Regressor.load(paths.PREDICTOR_DIR_PATH)
    logger.info("Making predictions...")
    predictions_df = predict_with_model(model, x_test)[[data_schema.id, 'prediction_label']]
    predictions_df.rename(columns={'prediction_label': 'prediction'}, inplace=True)

    print(predictions_df)

    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=paths.PREDICTIONS_FILE_PATH
    )

    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()
