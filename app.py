import os 
import sys 
from dotenv import load_dotenv
load_dotenv 

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
DATABASE_URL = f"mysql+mysqlconnector://{safe_user}:{safe_pass }@{MYSQL_HOST}/{MYSQL_DATABASE}"


engine = create_engine(DATABASE_URL,pool_pre_ping=True,pool_recycle=1800,future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine,future=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
        
app=FastAPI()
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)       

templates=Jinja2Templates(directory="./templates")



def table_exists(table_name: str) -> bool:
    """Check if the given table exists in the MySQL database."""
    try:
        insp = inspect(engine)
        return insp.has_table(table_name)
    except Exception as e:
        raise BankException(e, sys)
    
    
    
def read_table_as_dataframe(table_name:str)->pd.DataFrame:
    try:
        with engine.connect() as connection:
            query=f"SELECT * FROM {table_name};"
            df=pd.read_sql(query,connection)
            return df
    except Exception as e:
        raise BankException(e,sys)
    
    
def write_df(df:pd.DataFrame,table_name:str)->None:
    try:
        with engine.connect() as connection:
            df.to_sql(name=table_name,con=connection,if_exists='replace',index=False)
    except Exception as e:
        raise BankException(e,sys)
    


@app.get("/",tags=["authentication"])
async def index(request:Request):
    try:
        df = read_table_as_dataframe("predictions")
        table_html = df.to_html(classes="table table-striped")
    except Exception:
        table_html= "<p>No data yet.</p>"
    return templates.TemplateResponse("table.html",{"request":request, "table": table_html})

@app.get("/train",tags=["train"])
async def train_route(request:Request):
    try:
        training_pipeline=TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise BankException(e,sys)
    
    

@app.post("/predict",tags=["predict"])
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        # model_path=os.path.join("final_model","model.pkl")
        # preprocessor_path=os.path.join("final_model","preprocessor.pkl")
        
        model:BankModel=load_object("final_model/model.pkl")
        preprocessor=load_object("final_model/preprocessor.pkl")
        
        bank_model=BankModel(preprocessor=preprocessor,model=model)
        print(df.loc[0])
        
        y_pred=bank_model.predict(df)
        df["predicted_column"] = ["yes" if int(i) == 1 else "no" for i in y_pred]

        # Save and store
        df.to_csv("prediction_output/output.csv", index=False)
        write_df(df, "predictions")
        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse(
            "table.html",
            {"request": request, "message": "Prediction completed!", "table_html": table_html},
        )

    except Exception as e:
        raise BankException(e,sys)
    
if __name__=="__main__":
    from uvicorn import run
    run(app,host="0.0.0.0",port=8000)