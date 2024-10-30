from django.urls import path
from detect import views

urlpatterns = [
    path("detect/", views.detection),
    path("history/", views.history),
    path("comment/", views.upload_comment),
    path("clear/", views.clear),
]
