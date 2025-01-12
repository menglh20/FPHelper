from django.shortcuts import render
from common.models import Result, User
from django.http import JsonResponse
import datetime
import time
from detect.Download import DownloadImage
from detect.Detect import detect
import json
# Create your views here.


def detection(request):
    print("start detect")
    print(request)
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        name = body["name"]
        fileID = body["fileID"]
        print("fileID")
        print(fileID)
        # logger.info(f"[detect] Detection result: {name}")
        # if not User.objects.filter(name=name).exists():
        #     logger.info(f"[detect] User does not exist: {name}")
        #     return JsonResponse({
        #         "code": 400,
        #         "message": "User does not exist"
        #     })
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%Y.%m.%d %H:%M:%S")
        save_name = current_time.replace(".", "").replace(":", "")
        save_path = f"media/{name}_{save_name}/"
        # record = Result(name=name, result=0, detail="获取图片中", comment="", save_path=save_path, time=current_time)
        # record.save()

        downloadImage = DownloadImage()
        downloadImage.get(fileID, save_path)
        result, detail = detect(save_path)
        # record.result = result
        # record.detail = str(detail)
        # record.save()
        # logger.info(f"[detect] Detection success: {name} {result} {detail}")
        print("Detect Finished!")
        print(result)
        print(detail)
        return JsonResponse({
            "code": 200,
            "time": current_time,
            "result": result,
            "detail": detail,
        })
    else:
        return JsonResponse({
            "code": 400,
            "message": "Invalid request"
        })


def history(request):
    if request.method == "POST":
        name = request.POST.get("name")
        page = int(request.POST.get("page", 1))
        
        print(name + " is requesting history for page " + str(page))

        if not User.objects.filter(name=name).exists():
            return JsonResponse({
                "code": 400,
                "message": "User does not exist"
            })

        results = Result.objects.filter(name=name).order_by("-time")
        total = results.count()
        results = results[(page - 1) * 10:page * 10]

        return JsonResponse({
            "code": 200,
            "total": total,
            "page": page,
            "results": [{
                "id": result.id,
                "result": result.result,
                "time": result.time,
                "detail": result.detail,
                "comment": result.comment
            } for result in results]
        })
    else:
        return JsonResponse({
            "code": 400,
            "message": "Invalid request"
        })


def upload_comment(request):
    if request.method == "POST":
        id = request.POST.get("id")
        comment = request.POST.get("comment")
        
        print("Commenting on result " + id)
        
        if not Result.objects.filter(id=id).exists():
            return JsonResponse({
                "code": 400,
                "message": "Result does not exist"
            })

        record = Result.objects.filter(id=id).first()
        record.comment = comment
        record.save()

        return JsonResponse({
            "code": 200,
            "message": "Comment success"
        })
    else:
        return JsonResponse({
            "code": 400,
            "message": "Invalid request"
        })


# 内部接口
def clear(request):
    if request.method == "POST":
        secret_key = request.POST.get("secret_key")
        if secret_key != "123456":
            return JsonResponse({
                "code": 400,
                "message": "Invalid secret key"
            })
        Result.objects.all().delete()
    else:
        return JsonResponse({
            "code": 400,
            "message": "Invalid request"
        })
        

def get_all(request):
    if request.method == "POST":
        secret_key = request.POST.get("secret_key")
        if secret_key != "123456":
            return JsonResponse({
                "code": 400,
                "message": "Invalid secret key"
            })
        results = Result.objects.all().order_by("name")
        return JsonResponse({
            "code": 200,
            "results": [{
                "id": result.id,
                "name": result.name,
                "result": result.result,
                "time": result.time,
                "detail": result.detail,
                "comment": result.comment
            } for result in results]
        })
    else:
        return JsonResponse({
            "code": 400,
            "message": "Invalid request"
        })