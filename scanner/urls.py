from django.urls import path
from django.conf import settings 
from django.conf.urls.static import static
from scanner.views import *

urlpatterns = [
    path('image_upload', candi_image_scan, name = 'image_upload'), 
    path('success', success, name = 'success'), 
]

if settings.DEBUG: 
        urlpatterns += static(settings.MEDIA_URL, 
                              document_root=settings.MEDIA_ROOT) 