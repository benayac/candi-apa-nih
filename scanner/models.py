from django.db import models

# Create your models here.
class Candi(models.Model):
    candi_url = models.TextField(blank=True, null=True)
    candi_img = models.ImageField(upload_to='images/', blank=True, null=True)