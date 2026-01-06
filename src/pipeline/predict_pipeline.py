import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import ConfigurationManager
from src.utils import load_keras, load_npy, load_faiss, load_joblib

class PredictionPipeline:
    def __init__(self, embedding_model, pca, index, image_paths):
        self.embedding_model = embedding_model
        self.pca = pca
        self.index = index
        self.image_paths = image_paths

    def load_and_preprocess_image(self, data, img_size=(224,224)):
        if isinstance(data, str):  # file path
            img = tf.keras.preprocessing.image.load_img(data, target_size=img_size)
        else:  # BytesIO uploaded file
            img = Image.open(data)
            img = img.resize(img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        return np.expand_dims(img, axis=0)

    def initiate_pipeline(self, data, k):
        logging.info("Initiating prediction pipeline.")
        try:
            logging.info("Preprocessing image.")
            img = self.load_and_preprocess_image(data)

            embedding = self.embedding_model.predict(img, verbose=0)
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

            embedding_pca = self.pca.transform(embedding)

            _, indices = self.index.search(embedding_pca, k)
            return self.image_paths[indices[0]].tolist()

        except Exception as e:
            raise CustomException(e, sys)