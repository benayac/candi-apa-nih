from django.shortcuts import render, redirect
from django.http import HttpResponse
from . import forms
import numpy as np
import requests
from io import BytesIO
import json
from PIL import Image

# Create your views here.
def candi_image_scan(request):
    form = forms.CandiForm(request.POST, request.FILES)
    if form.is_valid():
        # if (form.cleaned_data.get('candi_img') is not None):
        #     img = form.cleaned_data.get('candi_img')
        #     text = make_prediction(img)
        if(form.cleaned_data.get('candi_url') is not None):
            data = image_process_from_url(form.cleaned_data.get('candi_url'))
            text = make_prediction(data)
        #form = forms.CandiForm()
        return render(request, 'index.html', {'form':form, 'code':text})
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
    img = Image.open(BytesIO(response.content))
    b, g, r = img.split()
    img = np.asarray(Image.merge("RGB", (r, g, b)))
    return img

def make_prediction(data):
    scoring_uri = "http://784740cf-b174-41c2-a514-547956c27618.eastus2.azurecontainer.io/score"
    input_data = json.dumps({"data":data.tolist()})
    headers = {'Content-Type':'application/json'}
    resp = requests.post(scoring_uri, input_data, headers=headers)
    prediction = json.loads(resp.text)["result"]
    class_names = ['Candi Borobudur', 'Candi Brahu', 'Candi Dieng', 'Candi Mendut', 'Candi Prambanan']
    return class_names[np.argmax(prediction[0])]