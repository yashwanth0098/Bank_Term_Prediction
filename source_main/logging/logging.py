import logging 
from datetime import datetime
import os

LOGFILE= f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'
LOGDIR= os.path.join(os.getcwd(), 'logs', LOGFILE)

os.makedirs(LOGDIR, exist_ok=True)

LOGFILEPATH= os.path.join(LOGDIR, LOGFILE)

logging.basicConfig(
    filename=LOGFILEPATH,
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  
    level=logging.INFO,
    

)