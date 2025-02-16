from django.urls import path
from detect import views

urlpatterns = [
    path("detect/", views.detection),
    path("detect_by_video/", views.detect_by_video),
    path("history/", views.history),
    path("comment/", views.upload_comment),
    path("clear/", views.clear),
    path("get_all/", views.get_all)
]
