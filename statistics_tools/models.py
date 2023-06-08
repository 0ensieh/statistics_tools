from django.db import models


class CleanedFile(models.Model):
    excel_file = models.FileField(upload_to='uploads/')
    cleaned_file = models.FileField(upload_to='uploads/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)