from django.shortcuts import render, redirect
from django.http import HttpResponse
from . import forms
import numpy as np
import requests
from io import BytesIO
import json

# Create your views here.
def candi_image_scan(request):
    form = forms.CandiForm(request.POST, request.FILES)
    if form.is_valid():
        img = form.cleaned_data.get('candi_img')
        data = image_process(form.cleaned_data.get('candi_url'))
        text = make_prediction(data)
        return render(request, 'show_img.html', {'code':text})
    else:
        form = forms.CandiForm()
        return render(request, 'index.html', {'form':form})

def success(request):
    return HttpResponse('Successfully uploaded')

def image_process(url):
    response = requests.get(url)
    img = np.asarray(Image.open(BytesIO(response.content)))
    return img

def make_prediction(data):
    scoring_uri = "http://3378d5ec-a23c-4078-83c9-7d1acf8f0a36.eastus2.azurecontainer.io/score"
    input_data = json.dumps({"data":data.tolist()})
    headers = {'Content-Type':'application/json'}
    resp = requests.post(scoring_uri, input_data, headers=headers)
    prediction = json.loads(resp.text)["result"]
    class_names = ['Candi Borobudur', 'Candi Brahu', 'Candi Dieng', 'Candi Mendut', 'Candi Prambanan']
    return class_names[np.argmax(prediction[0])]