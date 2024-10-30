from django.shortcuts import render
from detect.Detect import detect
from common.models import Result, User
from django.http import JsonResponse
import datetime
import time

# Create your views here.


def detection(request):
    if request.method == "POST":
        name = request.POST.get("name")
        pic_at_rest = request.FILES.get("pic_at_rest")
        pic_forehead_wrinkle = request.FILES.get("pic_forehead_wrinkle")
        pic_eye_closure = request.FILES.get("pic_eye_closure")
        pic_smile = request.FILES.get("pic_smile")
        pic_snarl = request.FILES.get("pic_snarl")
        pic_lip_pucker = request.FILES.get("pic_lip_pucker")

        if not User.objects.filter(name=name).exists():
            return JsonResponse({
                "code": 400,
                "message": "User does not exist"
            })

        start_t = time.time()

        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%Y.%m.%d %H:%M:%S")
        save_name = current_time.replace(".", "").replace(":", "")

        save_path = f"media/{name}_{save_name}/"

        with open(save_path + "pic_at_rest.jpg", "wb") as f:
            for chunk in pic_at_rest.chunks():
                f.write(chunk)
                
        with open(save_path + "pic_forehead_wrinkle.jpg", "wb") as f:
            for chunk in pic_forehead_wrinkle.chunks():
                f.write(chunk)
                
        with open(save_path + "pic_eye_closure.jpg", "wb") as f:
            for chunk in pic_eye_closure.chunks():
                f.write(chunk)
                
        with open(save_path + "pic_smile.jpg", "wb") as f:
            for chunk in pic_smile.chunks():
                f.write(chunk)
                
        with open(save_path + "pic_snarl.jpg", "wb") as f:
            for chunk in pic_snarl.chunks():
                f.write(chunk)
                
        with open(save_path + "pic_lip_pucker.jpg", "wb") as f:
            for chunk in pic_lip_pucker.chunks():
                f.write(chunk)

        record = Result(name=name, result=0, detail="", comment="检测中", save_path=save_path, time=current_time)
        record.save()

        try:
            result, detail = detect(save_path, debug=False)

            record.result = result
            record.detail = detail
            record.save()

            end_t = time.time()

            return JsonResponse({
                "code": 200,
                "time": round(end_t - start_t, 2),
                "result": result,
                "detail": detail,
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({
                "code": 500,
                "message": str(e)
            })
    else:
        return JsonResponse({
            "code": 400,
            "message": "Invalid request"
        })


def history(request):
    if request.method == "POST":
        name = request.POST.get("name")
        print(name)
        page = int(request.POST.get("page", 1))

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


def clear(request):
    if request.method == "POST":
        Result.objects.all().delete()
        return JsonResponse({
            "code": 200,
            "message": "Clear success"
        })