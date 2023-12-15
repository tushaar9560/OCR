import os
import sys
from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch



@dataclass
class PredictionPipelineConfig:
    model_path: str ='artifacts/model'
    model_name: str = 'Salesforce/blip-image-captioning-base'

class PredictionPipeline:
    def __init__(self):
        '''
        Initiate the prediction pipeline
        '''
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.processor = None
        self.model = None
        self.load_processor_model()
    
    def predict(self, input_image: Image) -> str:
        '''
        Takes image as an input and returns its caption
        Params:
            input_image: Image -Image uploaded by user
        Returns:
            caption: str
        '''
        try:
            logging.info('Prediction Pipeline Started')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = self.processor(images=input_image, return_tensors="pt").to(device)

            pixel_values = inputs.pixel_values

            generated_ids = self.model.generate(pixel_values= pixel_values, max_length=50)

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logging.info('Prediction Completed')

            return caption
        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)
        
    def load_processor_model(self):
        if os.path.exists(self.prediction_pipeline_config.model_path):
            self.processor = AutoProcessor.from_pretrained(self.prediction_pipeline_config.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.prediction_pipeline_config.model_path)

        else:
            self.processor = None
            self.model = None