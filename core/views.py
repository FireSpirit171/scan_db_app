import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UploadDatasetForm
from uuid import uuid4
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

DATA_STORE = {}

def index(request):
    return render(request, 'index.html')


def upload_dataset(request):
    if request.method == 'POST':
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            model_choices = form.cleaned_data['models']
            df = pd.read_csv(csv_file)

            # Название датасета без расширения
            dataset_name = os.path.splitext(csv_file.name)[0]
            dataset_dir = os.path.join(settings.MEDIA_ROOT, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            target_column = df.columns[-1]
            X = df.iloc[:, :-1]
            y = df[target_column]

            split_idx = int(len(df) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            results = {}

            def save_plot(y_true, y_pred, title, method_name):
                fig, ax = plt.subplots()
                ax.plot(range(100), y_true[:100].values, label='Истинное')
                ax.plot(range(100), y_pred[:100], label='Предсказание')
                ax.legend()
                ax.set_title(title)

                # Сохраняем в dataset_dir под названием модели
                img_name = f"{method_name}.png"
                img_path = os.path.join(dataset_dir, img_name)
                fig.savefig(img_path)
                plt.close(fig)
                return f"{dataset_name}/{img_name}"  # относительный путь для <img src>

            if 'linear' in model_choices:
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                img_rel_path = save_plot(y_test, preds, 'Линейная регрессия', 'linear')
                results['linear'] = {
                    'plot': img_rel_path,
                    'accuracy': r2_score(y_test, preds)
                }

            if 'tree' in model_choices:
                model = DecisionTreeRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                img_rel_path = save_plot(y_test, preds, 'Дерево решений', 'tree')
                results['tree'] = {
                    'plot': img_rel_path,
                    'accuracy': r2_score(y_test, preds)
                }

            if 'rf' in model_choices:
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                img_rel_path = save_plot(y_test, preds, 'Random Forest (ансамблевый)', 'rf')
                results['rf'] = {
                    'plot': img_rel_path,
                    'accuracy': r2_score(y_test, preds)
                }

            if 'boost' in model_choices:
                model = GradientBoostingRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                img_rel_path = save_plot(y_test, preds, 'Gradient Boosting (ансамблевый)', 'boost')
                results['boost'] = {
                    'plot': img_rel_path,
                    'accuracy': r2_score(y_test, preds)
                }

            DATA_STORE['results'] = results
            return redirect('results')
    else:
        form = UploadDatasetForm()
    return render(request, 'upload.html', {'form': form})


def results(request):
    results = DATA_STORE.get('results', {})
    return render(request, 'results.html', {'results': results})
