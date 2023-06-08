from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from datetime import datetime, timedelta    
from json import JSONEncoder
import scipy.stats as stats
import numpy as np
import pandas as pd


def is_normal_distribution(values):
    _, p_value = stats.normaltest(values)  # Normality test
    significance_level = 0.05  # Set the significance level for the test

    return p_value > significance_level



def is_uniform_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    min_value = min(values)
    max_value = max(values)

    if max_value - min_value + 1 != len(unique_values):
        return False

    # تبدیل داده‌ها به مقیاس [0, 1]
    scaled_values = [(val - min_value) / (max_value - min_value) for val in values]

    _, p_value = stats.kstest(scaled_values, 'uniform')  # آزمون کولموگروف-اسمیرنوف
    significance_level = 0.05  # سطح معناداری را برای آزمون تعیین کنید

    return p_value > significance_level


def is_bernoulli_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    _, p_value = stats.chisquare(values)  # Chi square test
    significance_level = 0.05  # Set the significance level for the test

    return p_value > significance_level


def is_binomial_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    n = len(values)
    p = sum(values) / n

    _, p_value = stats.binom_test(sum(values), n, p)  # binomial test
    significance_level = 0.05  # Set the significance level for the test

    return p_value > significance_level


def is_negative_binomial_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    p = sum(values) / len(values)

    _, p_value = stats.nbinom_test(sum(values), len(values), p)  # negative binomial test
    significance_level = 0.05  # Set the significance level for the test

    return p_value > significance_level


def is_geometric_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    p = 1 / (sum(values) / len(values))

    _, p_value = stats.geom_test(max(values), p)  # geometric test
    significance_level = 0.05  # Set the significance level for the test

    return p_value > significance_level


def is_superior_geometric_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    max_value = max(values)
    p = 1 / (sum(values) / len(values))

    # superior geometric test
    if max_value == 1:
        return True

    for i in range(2, max_value+1):
        p *= 1 / (sum(values) / len(values))
        if p < 0.5:
            return False

    return True


def is_poisson_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    lambda_ = sum(values) / len(values)

    _, p_value = stats.kstest(values, 'poisson', args=(lambda_,))  # Kolmogorov Smirnov test
    significance_level = 0.05  # Set the significance level for the test

    return p_value > significance_level


def is_discrete_uniform_distribution(values):
    unique_values = set(values)
    if len(unique_values) > 2:
        return False

    min_value = min(values)
    max_value = max(values)

    if max_value - min_value + 1 != len(unique_values):
        return False

    return True


def is_beta_distribution(values):
    min_value = min(values)
    max_value = max(values)

    # Check if any value is outside the valid range
    if min_value <= 0 or max_value >= 1:
        return False

    # Normalize data to (0, 1) range
    normalized_values = (values - min_value) / (max_value - min_value)

    alpha, beta, loc, scale = stats.beta.fit(normalized_values, floc=0, fscale=1)
    _, p_value = stats.kstest(normalized_values, 'beta', args=(alpha, beta, loc, scale))
    significance_level = 0.05

    return p_value > significance_level


def is_exponential_distribution(values):
    _, p_value = stats.kstest(values, 'expon')  
    significance_level = 0.05  

    return p_value > significance_level


def is_gamma_distribution(values):
    mean = np.mean(values)
    variance = np.var(values)
    shape = (mean ** 2) / variance
    scale = variance / mean

    _, p_value = stats.kstest(values, 'gamma', args=(shape, 0, scale))  
    significance_level = 0.05 

    return p_value > significance_level


def is_weibull_distribution(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    c = (std / mean) ** -1.086
    
    _, p_value = stats.kstest(values, 'weibull_min', args=(c,))  
    significance_level = 0.05  

    return p_value > significance_level



def is_pareto_distribution(values):
    min_value = np.min(values)
    b = 1 + len(values) / np.sum(np.log(values / min_value))
    loc = min_value

    _, p_value = stats.kstest(values, 'pareto', args=(b, loc))  
    significance_level = 0.05  

    return p_value > significance_level


def is_lognormal_distribution(values):
    shape, loc, scale = stats.lognorm.fit(values, floc=0)
    _, p_value = stats.kstest(values, 'lognorm', args=(shape, loc, scale))  
    significance_level = 0.05  

    return p_value > significance_level


def is_logistic_distribution(values):
    _, p_value = stats.kstest(values, 'logistic')  
    significance_level = 0.05  

    return p_value > significance_level


def is_cauchy_distribution(values):
    _, p_value = stats.kstest(values, 'cauchy') 
    significance_level = 0.05  

    return p_value > significance_level


def determine_distribution_type(data):
    distribution_types = {}

    for column in data.columns:
        values = data[column].dropna()

        if values.dtype.kind not in 'bifc':  # Skip non-numeric columns
            continue
        
        values = values.astype(float)
        

        if is_normal_distribution(values):
            distribution_types[column] = 'توزیع نرمال'
        elif is_uniform_distribution(values):
            distribution_types[column] = 'توزیع یکنواخت'
        elif is_bernoulli_distribution(values):
            distribution_types[column] = 'توزیع برنولی'
        elif is_binomial_distribution(values):
            distribution_types[column] = 'توزیع دو جمله ای'
        elif is_negative_binomial_distribution(values):
            distribution_types[column] = 'توزیع دوجمله ای منفی'
        elif is_geometric_distribution(values):
            distribution_types[column] = 'توزیع هندسی'
        elif is_superior_geometric_distribution(values):
            distribution_types[column] = 'توزیع فوق هندسی'
        elif is_poisson_distribution(values):
            distribution_types[column] = 'توزیع پواسون'
        elif is_discrete_uniform_distribution(values):
            distribution_types[column] = 'توزیع یکنواخت گسسته'
        elif is_beta_distribution(values):
            distribution_types[column] = 'توزیع بتا'
        elif is_exponential_distribution(values):
            distribution_types[column] = 'توزیع نمایی'
        elif is_gamma_distribution(values):
            distribution_types[column] = 'توزیع گاما'
        elif is_weibull_distribution(values):
            distribution_types[column] = 'توزیع وایبل'
        elif is_pareto_distribution(values):
            distribution_types[column] = 'توزیع پارتو'
        elif is_lognormal_distribution(values):
            distribution_types[column] = 'توزیع لگ نرمال'
        elif is_logistic_distribution(values):
            distribution_types[column] = 'توزیع لجستیک'
        elif is_cauchy_distribution(values):
            distribution_types[column] = 'توزیع خی دو'
        else:
            distribution_types[column] = 'توزیع نامعلوم'

    return distribution_types


def prepare_chart_data(data):
    chart_data = []
    for column in data.columns:
        values = data[column].tolist()
        chart_data.append({'name': column, 'data': values})
    return chart_data


class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.int64):
            return int(obj)  # Convert int64 to int
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X)


def logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model.predict(X)


def decision_tree(X, y):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    return model.predict(X)


def random_forest(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model.predict(X)


def knn(X, y):
    model = KNeighborsRegressor()
    model.fit(X, y)
    return model.predict(X)


def pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    model = LinearRegression()
    model.fit(X_pca, y)
    return model.predict(X_pca)

def arima_forecast(X, y):
    # Fit the ARIMA model
    model = ARIMA(y, order=(1, 0, 0))  # Adjust the order as per your requirement
    model_fit = model.fit()

    # Forecast future values
    future_dates = pd.date_range(start=X.index[-1], periods=30, freq='D')
    forecast = model_fit.predict(start=len(y), end=len(y) + 29)  # Adjust the forecast range as per your requirement

    return forecast