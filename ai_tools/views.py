from django.shortcuts import render
from .forms import ImageToTextForm
from .models import ImageConvert
from PIL import Image
import pytesseract


def image_to_text_converter(request):
    context = {}
    if request.method == 'POST':
        form = ImageToTextForm(request.POST, request.FILES)
        if form.is_valid():
            languages = ['en', 'fa', 'digits']
            image = form.cleaned_data.get('image')
            obj = ImageConvert.objects.create(image=image)
            image = Image.open(obj.image.path)
            image = image.convert('L') #convert to grayscale
            image = image.point(lambda x: 0 if x < 128 else 255, "1")
            text = pytesseract.image_to_string(image)
            text = pytesseract.image_to_string(image, lang='fas+eng+num')

            context['form'] = form
            context['text'] = text
            return render(request, 'ai_tools/image_to_text_converter.html', context)
    else:
        form = ImageToTextForm()
        context['form'] = form    
    return render(request, 'ai_tools/image_to_text_converter.html', context)