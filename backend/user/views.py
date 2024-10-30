from django.shortcuts import render
from common.models import User
from django.http import HttpResponse, JsonResponse
import re

# Create your views here.

def is_chinese_mobile_number(text):
    pattern = r"^1[3-9]\d{9}$"
    
    if re.match(pattern, text):
        return True
    else:
        return False


def register(request):
    if request.method == "POST":
        name = request.POST.get("name")
        password = request.POST.get("password")

        if User.objects.filter(name=name).exists():
            return JsonResponse({
                "code": 400,
                "message": "User already exists"
            })
            
        

        User.objects.create(name=name, password=password)
        return JsonResponse({
            "code": 200,
            "message": "User created"
        })
    else:
        return JsonResponse({
            "code": 400,
            "message": "Invalid request"
        })


def login(request):
    if request.method == "POST":
        name = request.POST.get("name")
        password = request.POST.get("password")

        if not User.objects.filter(name=name).exists():
            print(name, password)
            return JsonResponse({
                "code": 400,
                "message": "Login failed"
            })

        user = User.objects.get(name=name)
        if user.password != password:
            return JsonResponse({
                "code": 401,
                "message": "Login failed"
            })

        return JsonResponse({
            "code": 200,
            "message": "Login successful"
        })
    else:
        return JsonResponse({
            "code": 402,
            "message": "Invalid request"
        })
