from source_main.constants.pipelineconstants import MODEL_FILE_NAME,SAVE_MODEL_DIR
from source_main.exception.exception import BankException
from source_main.logging.logging import logging
import os
import sys
from sklearn.pipeline import Pipeline
import pandas as pd

class BankModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise BankException(e,sys)
        
    def predict(self,X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # If model is a Pipeline (has transformers), DON'T pre-transform
        if isinstance(self.model, Pipeline):
            return self.model.predict(X)

        # Otherwise, use the saved preprocessor first
        X_t = self.preprocessor.transform(X)
        return self.model.predict(X_t)