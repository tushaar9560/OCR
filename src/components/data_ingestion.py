import gdown
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
from zipfile import ZipFile


@dataclass
class DataIngestionConfig:
    # Google Drive ID of the file to be downloaded from google drive and saved in local machine as a csv file
    data_path: str
    gdrive_url: str = "https://drive.google.com/uc?id=1IX-gcZVnTvIXSv4DxW7MEc3ys3XMuyUU"
    download_file_path: str


class DataIngestion:
    def __init__(self, data_path: str, download_file_path: str):
        '''
        Intializes the data ingestion class with defined download and extract path
        Params:
            download_file_path: str - Path to download file
            data_path: str - Path to extract file
        '''
        self.ingestion_config = DataIngestionConfig(data_path=data_path,
                                                    download_file_path=download_file_path)
        
    def download_data(self):
        '''
        Downloads the data zip file from gdrive and saves it locally in
        download_file_path
        Return:
            download_file_path: str
        '''
        try:
            logging.info('Data is downloading...')
            gdown.download(self.ingestion_config.gdrive_url,
                           self.ingestion_config.download_file_path)
            logging.info('Download finished <3')

            return self.ingestion_config.download_file_path

        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)
        
    def unzip_data(self):
        '''
        Extract the downloaded data zipfile and saves it in data_path
        Return:
            data_path: str
        '''
        try:
            logging.info('Extraction started...')
            with ZipFile(self.ingestion_config.download_file_path, 'r') as file:
                file.extractall(self.ingestion_config.data_path)
            logging.info('Extraction done!!!!')
            return self.ingestion_config.data_path
        
        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)