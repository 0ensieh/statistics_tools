from django import forms
from .models import CleanedFile


class DataDistributionForm(forms.Form):
    excel_file = forms.FileField(label='فایل اکسل خود را انتخاب کنید', widget=forms.FileInput(attrs={'class': 'form-control mt-2'}))


class PredictDataForm(forms.Form):
    file = forms.FileField(label='فایل اکسل خود را انتخاب کنید', widget=forms.FileInput(attrs={'class': 'form-control my-3 mt-2'}))
    target_column = forms.CharField(label='نام ستون مد نظر برای پیش بینی را وارد کنید', widget=forms.TextInput(attrs={'class': 'form-control my-3 mt-2', 'dir': 'ltr'}))


class CleanDataForm(forms.ModelForm):
    class Meta:
        model = CleanedFile
        fields = ['excel_file']
        labels = {'excel_file': 'فایل اکسل خود را انتخاب کنید'}
        widgets = {
            'excel_file': forms.FileInput(attrs={'class': 'form-control mt-2'})
            
        }