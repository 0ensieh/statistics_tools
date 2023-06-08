from django import forms


class ImageToTextForm(forms.Form):
    image = forms.ImageField(widget=forms.FileInput(attrs={'class': 'form-control mt-2 mb-5'}), label='انتخاب عکس')