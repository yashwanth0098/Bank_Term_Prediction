from source_main.logging import logging
import os 
import sys

class BankException(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message= error_message
        _,_,exc_tb= error_details.exc_info()
        self.file_name= exc_tb.tb_frame.f_code.co_filename 
        self.line_number= exc_tb.tb_lineno
        
        
        
    def __str__(self):
        return  "Error occurred in script[{0}] at line number[{1}] error message[{2}]".format(
            self.file_name,
            self.line_number,
            str(self.error_message)
        )