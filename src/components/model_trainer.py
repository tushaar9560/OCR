from transformers import AutoProcessor, BlipForConditionalGeneration
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ModelTrainerConfig:
    data_path: str
    model_path: str
    model_name: str = "Salesforce/blip-image-captioning-base"
    

class ModelTrainer:
    def __init__(self, data_path:str, model_path:str):
        '''
        Initiate the model training class to load the data
        and train the model
        Params:
            data_path: str - Path of training data
            model_path: str - Path to save model
        '''
        self.model_trainer_config = ModelTrainerConfig(data_path=data_path, model_path=model_path)
        self.dataset = None
        self.processor = None
        # self.model = None
    
    def load_dataset(self):
        '''
        Load the dataset from the data_path
        '''
        try:
            logging.info('Loading dataset...')
            self.dataset = load_dataset('imagefolder', data_dir=self.model_trainer_config.data_path, split='train')
            self.processor = AutoProcessor.from_pretrained(self.model_trainer_config.model_name)
            self.dataset = ImageCaptioningDataset(self.dataset, self.processor)
            logging.info('Successfully Loaded')

        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)
    
    def train_model(self):
        '''
        Training the model
        Return:
            model_path: str - Path where model is saved
        '''
        try:
            logging.info('Training the model...')
            train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=4)
            model = BlipForConditionalGeneration.from_pretrained(self.model_trainer_config.model_name)

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            model.train()

            for epoch in range(1):
                logging.info("Epoch: ", epoch)
                for idx, batch in enumerate(train_dataloader):
                    input_ids = batch.pop("input_ids").to(device)
                    pixel_values = batch.pop("pixel_values").to(device)

                    outputs = model(input_ids=input_ids,
                                    pixel_values=pixel_values,
                                    labels=input_ids)
                    
                    loss = outputs.loss
                    if idx%100==0:
                        print("Index: ", idx, "Loss:", loss.item())

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
            
            model.save_pretrained(self.model_trainer_config.model_path)
            logging.info('Training Completed!!!')
            return self.model_trainer_config.model_path
        
        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding