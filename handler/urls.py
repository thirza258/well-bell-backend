from django.urls import path
from handler.views import AskRAG, FormDataAPIView

urlpatterns = [
    path('ask/', AskRAG.as_view(), name='ask'),
     path('personalized/', FormDataAPIView.as_view(), name='form-data'),
]