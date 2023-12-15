from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from pipeline.prediction_pipeline import PredictionPipeline
from utils import check_model_exist
import os
import requests


class MyFlask(Flask):
    def run(self, host=None, port = None, debug = None, load_dotenv=None, **kwargs):
        if not self.debug or os.getenv('WERKZEUG_RUN_PATH') == 'true':
            with self.app_context():
                global predict_pipeline
                if predict_pipeline is None:
                    predict_pipeline = PredictionPipeline()
        
        super(MyFlask, self).run(host = host, port = port,
                                 debug=debug, load_dotenv=load_dotenv, **kwargs)
                


app = MyFlask(__name__)
predict_pipeline = None


@app.route('/', methods = ['GET', 'POST'])
def predict():
    global predict_pipeline
    if request.method=='GET':
        return render_template('predict.html', trained = check_model_exist(
            predict_pipeline.prediction_pipeline_config.model_path
        ),
        caption = None)
    

    else:
        if predict_pipeline.model is None:
            predict_pipeline.load_processor_model()
        
        os.makedirs('artifacts/inputs', exist_ok=True)
        image_file = request.files['Image']
        image_file_path = os.path.join('artifacts','inputs' , secure_filename(image_file.filename))
        image_file.save(image_file_path)

        image = Image.open(image_file_path).convert('RGB')

        text = predict_pipeline.predict(image)

        return render_template('predict.html', text=text, trained =check_model_exist(
            predict_pipeline.prediction_pipeline_config.model_path
        ))


if __name__ == '__main__':
    app.run('0.0.0.0', debug=False)