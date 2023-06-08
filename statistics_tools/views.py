from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from django.shortcuts import render, HttpResponse
from .forms import DataDistributionForm, PredictDataForm, CleanDataForm
from .utils import determine_distribution_type, prepare_chart_data, DateTimeEncoder, linear_regression, logistic_regression, decision_tree, random_forest, knn, pca, arima_forecast
import pandas as pd
import numpy as np
from json import dumps
from .models import CleanedFile


def data_distribution(request):
    if request.method == 'POST':
        form = DataDistributionForm(request.POST, request.FILES)
        file = request.FILES['excel_file']
        df = pd.read_excel(file)

        # Perform data analysis and determine distribution type
        distribution_types = determine_distribution_type(df)

        # Prepare data for Highcharts
        chart_data = prepare_chart_data(df)

        # Pass data to Highcharts view
        return render(request, 'statistics_tools/data_distribution.html', {'form': form, 'data': chart_data, 'distribution_types': distribution_types})
    else:
        form = DataDistributionForm()

    return render(request, 'statistics_tools/data_distribution.html', {'form': form})




def predict_data(request):
    if request.method == 'POST':
        form = PredictDataForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            target_column = form.cleaned_data['target_column']

            try:
                df = pd.read_excel(file)

                # Check if the target column exists in the DataFrame
                if target_column not in df.columns:
                    return HttpResponse("Target column not found in the dataset.")

                # Perform data preprocessing
                df_numeric = df.select_dtypes(include=['float', 'int']).dropna()

                # Separate the features (X) and target variable (y)
                X = df_numeric.drop(target_column, axis=1)
                y = df_numeric[target_column]

                if X.empty or y.empty:
                    return HttpResponse("Insufficient data for prediction.")

                # Perform predictions using different models
                models = {
                    'رگرسیون خطی': linear_regression,
                    'رگرسیون لجستیک': logistic_regression,
                    'درخت تصمیم گیری': decision_tree,
                    'جنگل های تصمیم تصادفی': random_forest,
                    'k-نزدیک ترین همسایگی': knn,
                    'PCA': pca, 
                    'پیش‌بینی ARIMA': arima_forecast, 
                }

                # Generate future dates starting from the current date
                current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                future_dates = [current_date + relativedelta(days=i) for i in range(1, 31)]

                # Prepare the data for JavaScript
                data = {}
                for model_name, model_fn in models.items():
                    predictions = model_fn(X, y)
                    data[model_name] = list(zip(future_dates, predictions))

                data = dumps(data, cls=DateTimeEncoder)

                context = {
                    'form': form,
                    'target_column': target_column,
                    'data': data,
                }

                return render(request, 'statistics_tools/predictions.html', context)

            except Exception as e:
                return HttpResponse(f"An error occurred: {str(e)}")
    else:
        form = PredictDataForm()
    return render(request, 'statistics_tools/predict_form.html', {'form': form})




def clean_data(request):
    if request.method == 'POST':
        form = CleanDataForm(request.POST, request.FILES)
        if form.is_valid():
            cleaned_file = form.save()
            excel_file_path = cleaned_file.excel_file.path

            df = pd.read_excel(excel_file_path)
            
            # cleaning duplicate records
            df.drop_duplicates(inplace=True)
            
            # cleaning records with na field(s)
            df.dropna(inplace=True)
            
            # saving cleaned data to new file
            cleaned_file_path = 'media/' + cleaned_file.excel_file.name

            df.to_excel(cleaned_file_path, index=False)
            
            cleaned_file.cleaned_file = cleaned_file_path
            cleaned_file.save()

            # return cleaned file to user
            with open(cleaned_file_path, 'rb') as file:
                response = HttpResponse(file.read(), content_type='application/vnd.ms-excel')
                response['Content-Disposition'] = 'attachment; filename=cleaned_file.xlsx'
                return response
    else:
        form = CleanDataForm()

    context = {'form': form}
    return render(request, 'statistics_tools/clean_data.html', context)