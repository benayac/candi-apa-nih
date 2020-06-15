from django.shortcuts import render, redirect
from django.http import HttpResponse
from . import forms
import numpy as np
import requests
from io import BytesIO
import json
from PIL import Image
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Create your views here.
def candi_image_scan(request):
    form = forms.CandiForm(request.POST, request.FILES)
    if form.is_valid():
        if(form.cleaned_data.get('candi_url') is not None):
            data = image_process_from_url(form.cleaned_data.get('candi_url'))
            text, confidence = make_prediction(data)
        return render(request, 'index.html', {'form':form, 'code':text, 'confidence':confidence})
    else:
        form = forms.CandiForm()
        return render(request, 'index.html', {'form':form})

def success(request):
    return HttpResponse('Successfully uploaded')

def image_process_file(img_file):
    img = Image.open(img_file.file)
    b, g, r = img.split()
    img = np.asarray(Image.merge("RGB"), (r, g, b))
    return img

def image_process_from_url(url):
    response = requests.get(url)
    img = img_to_array(Image.open(BytesIO(response.content)).resize((224, 224)))
    img = preprocess_input(img)
    img = np.array(img, dtype='float32')
    return img

def make_prediction(data):
    scoring_uri = "http://18385e05-db8d-4ef7-8d30-a6aaf2f3aa7d.eastus2.azurecontainer.io/score"
    input_data = json.dumps({"data":data.tolist()})
    headers = {'Content-Type':'application/json'}
    resp = requests.post(scoring_uri, input_data, headers=headers)
    prediction = json.loads(resp.text)["result"]
    class_names = ['Banyunibo', 'Borobudur', 'Brahu', 'Cangkuang', 'Dieng', 'Jabung', 'Jago', 
    'Kalasan', 'Mendut', 'Muara Takus', 'Padas', 'Pawon', 'Prambanan', 'Sambisari', 'Sari', 'Sewu']
    confidence = int(max(prediction)*100)
    return class_names[np.argmax(prediction)], confidence