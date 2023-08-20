from PIL import Image
import requests
from django.http import HttpResponse
from django.shortcuts import render
import torch
import sys
from image_generator.inference import run_inference

def generate_image(request):
    prompt = request.GET.get('text_input')
    image_url = ''
    if prompt:
        run_inference(prompt)
        image_url = "0.png"
    context = {'image_url': image_url}
    return render(request, 'generate.html', context)