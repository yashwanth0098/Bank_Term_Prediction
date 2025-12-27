import os 
import sys 
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text,inspect
from sqlalchemy.orm import sessionmaker

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from fastapi.responses import Response,RedirectResponse
from fastapi.templating import Jinja2Templates


import pandas as pd 
from source_main.exception.exception import BankException
from source_main.logging.logging import logging
from source_main.pipeline.training_pipeline_aws import TrainingPipeline
from source_main.utlis.main_utlis.utlis import load_object
from source_main.utlis.model.estimator import BankModel
from urllib.parse import quote_plus as urlquote


MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

safe_user = urlquote(MYSQL_USER)
safe_pass = urlquote(MYSQL_PASSWORD)

DATABASE_URL = (
    f"mysql+mysqlconnector://{safe_user}:{safe_pass}@{MYSQL_HOST}/{MYSQL_DATABASE}"
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

os.makedirs(os.path.join(BASE_DIR, "prediction_output"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "final_model"), exist_ok=True)


def table_exists(table_name: str) -> bool:
    try:
        insp = inspect(engine)
        return insp.has_table(table_name)
    except Exception as e:
        raise BankException(e, sys)

def read_table_as_dataframe(table_name: str) -> pd.DataFrame:
    try:
        with engine.connect() as connection:
            query = f"SELECT * FROM {table_name};"
            return pd.read_sql(query, connection)
    except Exception as e:
        raise BankException(e, sys)

def write_df(df: pd.DataFrame, table_name: str) -> None:
    try:
        with engine.connect() as connection:
            df.to_sql(
                name=table_name,
                con=connection,
                if_exists="replace",
                index=False,
            )
    except Exception as e:
        raise BankException(e, sys)




@app.get("/")
async def health():
    return {"status": "FastAPI is running"}

@app.get("/train", tags=["train"])
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise BankException(e, sys)

@app.post("/predict", tags=["predict"])
async def predict_route(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        model_path = os.path.join(BASE_DIR, "final_model", "model.pkl")
        preprocessor_path = os.path.join(BASE_DIR, "final_model", "preprocessor.pkl")

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            return Response("Model files not found", status_code=500)

        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)

        bank_model = BankModel(
            preprocessor=preprocessor,
            model=model,
        )

        y_pred = bank_model.predict(df)
        df["predicted_column"] = [
            "yes" if int(i) == 1 else "no" for i in y_pred
        ]

        output_path = os.path.join(
            BASE_DIR, "prediction_output", "output.csv"
        )
        df.to_csv(output_path, index=False)

        write_df(df, "predictions")

        return {"message": "Prediction completed successfully"}

    except Exception as e:
        raise BankException(e, sys)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)