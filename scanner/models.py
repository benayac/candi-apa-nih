from django.db import models

# Create your models here.
class Candi(models.Model):
    candi_url = models.CharField(max_length=512)