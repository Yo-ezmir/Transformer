from django.contrib import admin
from django.urls import path, include
from transform import urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('transform.urls'))
]