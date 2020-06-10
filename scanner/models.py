from django.db import models

# Create your models here.
class Candi(models.Model):
    candi_url = models.TextField()
    candi_img = models.ImageField(upload_to='images/')