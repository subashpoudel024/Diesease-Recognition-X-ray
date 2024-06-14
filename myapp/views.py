import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your models
pneumonia_model = load_model('D:\DATA_SCIENCE\DiseasaeRecognitionLite\Models\pneumonia-model.h5')
tb_model = load_model('D:\DATA_SCIENCE\DiseasaeRecognitionLite\Models\_tb-model.h5')

def index(request):
    return render(request, 'myapp/index.html')

def predict(request):
    if request.method == 'POST' and request.FILES['xray_image']:
        xray_image = request.FILES['xray_image']
        fs = FileSystemStorage()
        filename = fs.save(xray_image.name, xray_image)
        uploaded_file_url = fs.url(filename)
        
        img_path = os.path.join(fs.location, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict with pneumonia model
        pneumonia_pred = pneumonia_model.predict(img_array)
        pneumonia_result = 'Pneumonia seems to be positive. Seek medical advice.' if pneumonia_pred[0][0] > 0.5 else 'Pneumonia seems to be negative. Person is safe.'
        pneumonia_class = 'positive' if pneumonia_pred[0][0] > 0.5 else 'negative'

        # Predict with TB model
        tb_pred = tb_model.predict(img_array)
        tb_result = 'TB seems to be positive. Seek medical advice.' if tb_pred[0][0] > 0.5 else 'TB seems to be negative. Person is safe.'
        tb_class = 'positive' if tb_pred[0][0] > 0.5 else 'negative'

        prediction = {
            'pneumonia': pneumonia_result,
            'pneumonia_class': pneumonia_class,
            'tb': tb_result,
            'tb_class': tb_class
        }

        fs.delete(filename)
        return render(request, 'myapp/index.html', {'prediction': prediction})

    return render(request, 'myapp/index.html')
