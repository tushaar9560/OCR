import os
import sys
from exception import CustomException
from utils import check_model_exist
from dataclasses import dataclass
from logger import logging 
# Load model directly
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


from PIL import Image
import torch


@dataclass
class PredictionPipelineConfig:
    model_path: str ='artifacts/model'
    model_name: str = "microsoft/trocr-base-printed"


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

            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.config.vocab_size = self.model.config.decoder.vocab_size
            self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
            self.model.config.max_length = 64
            self.model.config.early_stopping = True
            self.model.config.no_repeat_ngram_size = 3
            self.model.config.length_penalty = 2.0
            self.model.config.num_beams = 4

            generated_ids = self.model.generate(pixel_values= pixel_values)

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logging.info('Prediction Completed')

            return caption
        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)
        
    def load_processor_model(self):
        if os.path.exists(self.prediction_pipeline_config.model_path):
            self.processor = TrOCRProcessor.from_pretrained(self.prediction_pipeline_config.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.prediction_pipeline_config.model_path) 

        else:
            self.processor = None
            self.model = None