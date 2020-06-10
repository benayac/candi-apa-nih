from django.http import HttpResponse 
from django.shortcuts import render, redirect 
import requests
import json

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")