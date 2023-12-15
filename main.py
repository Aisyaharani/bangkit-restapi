import os
import uvicorn
import traceback
import tensorflow as tf

import numpy as np

from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector


from utils import load_image_into_numpy_array

model = tf.keras.models.load_model('EfficientNetB0_classifier.h5')

labels = ['aloevera',
          'banana',
          'bilimbi',
          'cantaloupe',
          'cassava',
          'coconut',
          'corn',
          'cucumber',
          'curcuma',
          'eggplant',
          'galangal',
          'ginger',
          'guava',
          'kale',
          'longbeans',
          'mango',
          'melon',
          'orange',
          'paddy',
          'papaya',
          'peper chili',
          'pineapple',
          'pomelo',
          'shallot',
          'soybeans',
          'spinach',
          'sweet potatoes',
          'tobacco',
          'waterapple',
          'watermelon']

app = FastAPI()

origins = ["http://localhost:8080"]  # Atur origins sesuai kebutuhan aplikasi Anda.

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "explora",
}

connection = mysql.connector.connect(**db_config)

def execute_query(query):
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

@app.get("/tanaman")
def get_tanaman():
    query = "SELECT * FROM tanaman"
    results = execute_query(query)
    return results

tanaman_data = get_tanaman()


@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        # Checking if it's an image
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        img = load_image_into_numpy_array(uploaded_file.file.read())

        results = model.predict(img)
        results = results[0]
        likely_class = np.argmax(results)
        confidence_score = float(results[likely_class])

        
        predicted_class_index = int(likely_class)
        predicted_class = labels[predicted_class_index]

        
        predicted_class = labels[int(likely_class)]

        return predicted_class
    
        # if tanaman_data: 
        #     predicted_tanaman_info = tanaman_data[predicted_class]
        #     predicted_class_info ={
        #         "predicted_class":predicted_class,
        #         "confidence_score": confidence_score,
        #         "tanaman_info": predicted_tanaman_info
        #     }
        #     return predicted_class_info
        # else:
        #     return {
        #         "predicted_class": predicted_class,
        #         "confidence_score": confidence_score,
        #         "tanaman_info": None
        #     }
        
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


@app.get("/quiz")
def get_quiz():

    query = "SELECT * FROM quiz"
    results = execute_query(query)
    return results



# Starting the server
# Your can check the API documentation easily using /docs after the server is running
if __name__ == "__main__":
    # Starting the server
    # Your can check the API documentation easily using /docs after the server is running
    port = os.environ.get("PORT", 8080)
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port)
