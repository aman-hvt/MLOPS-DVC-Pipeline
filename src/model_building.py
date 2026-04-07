import pandas as pd
import logging
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

logs_dir = 'Logs'
os.makedirs(logs_dir,exist_ok=True)

logger =  logging.getLogger('model-training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

Log_file_path = os.path.join(logs_dir,'model_training.log')
file_handler = logging.FileHandler(Log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train:np.ndarray,Y_train:np.ndarray, params:dict)-> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param Y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        logger.debug("Initilizing RandomForestClassifier with parameters: %s", params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug('Model training startede with %d samples', X_train.shape[0])
        clf.fit(X_train, Y_train)
        logger.debug('Model training completed')
        return clf
    except ValueError as e:
        logger.error("Value error during model training: %s", e)
        raise
    except Exception as e:
        logger.error("An error occurred during model training: %s", e)
        raise

def save_model(model,file_path:str)-> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('An error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = {'n_estimators': 25, 'random_state': 2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()