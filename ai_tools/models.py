from django.db import models

class ImageConvert(models.Model):
    image = models.ImageField(upload_to='images/')