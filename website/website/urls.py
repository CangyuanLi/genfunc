from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('randfunc/', include('randfunc.urls')),
    path('admin/', admin.site.urls),
]

