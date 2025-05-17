import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4
from datetime import datetime
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UploadDatasetForm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

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
            dataset_name = os.path.splitext(csv_file.name)[0]
            dataset_dir = os.path.join(settings.MEDIA_ROOT, 'models', dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            target_column = df.columns[-1]
            X = df.iloc[:, :-1]
            y = df[target_column]

            split_idx = int(len(df) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            results = {}

            def save_plot(y_true, y_pred, title, name):
                fig, ax = plt.subplots()
                ax.plot(range(100), y_true[:100].values, label='Истинное')
                ax.plot(range(100), y_pred[:100], label='Предсказание')
                ax.legend()
                ax.set_title(title)
                img_path = os.path.join(dataset_dir, f"{name}.png")
                fig.savefig(img_path)
                plt.close(fig)
                return f"models/{dataset_name}/{name}.png"

            if 'linear' in model_choices:
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['linear'] = {
                    'plot': save_plot(y_test, preds, 'Линейная регрессия', 'linear'),
                    'accuracy': r2_score(y_test, preds)
                }

            if 'tree' in model_choices:
                model = DecisionTreeRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['tree'] = {
                    'plot': save_plot(y_test, preds, 'Дерево решений', 'tree'),
                    'accuracy': r2_score(y_test, preds)
                }

            if 'rf' in model_choices:
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['rf'] = {
                    'plot': save_plot(y_test, preds, 'Random Forest', 'rf'),
                    'accuracy': r2_score(y_test, preds)
                }

            if 'boost' in model_choices:
                model = GradientBoostingRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['boost'] = {
                    'plot': save_plot(y_test, preds, 'Gradient Boosting', 'boost'),
                    'accuracy': r2_score(y_test, preds)
                }

            if 'knn' in model_choices:
                model = KNeighborsRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['knn'] = {
                    'plot': save_plot(y_test, preds, 'K-ближайших соседей', 'knn'),
                    'accuracy': r2_score(y_test, preds)
                }

            if 'svr' in model_choices:
                model = SVR()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['svr'] = {
                    'plot': save_plot(y_test, preds, 'Support Vector Regression', 'svr'),
                    'accuracy': r2_score(y_test, preds)
                }

            if 'lasso' in model_choices:
                model = Lasso()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['lasso'] = {
                    'plot': save_plot(y_test, preds, 'Лассо-регрессия', 'lasso'),
                    'accuracy': r2_score(y_test, preds)
                }

            if 'bagging' in model_choices:
                model = BaggingRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results['bagging'] = {
                    'plot': save_plot(y_test, preds, 'Bagging Regressor', 'bagging'),
                    'accuracy': r2_score(y_test, preds)
                }

            history_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
            history_dir = os.path.join(settings.BASE_DIR, 'history', history_id)
            os.makedirs(history_dir, exist_ok=True)

            info = {
                'dataset_name': dataset_name,
                'datetime': datetime.now().isoformat(),
                'method': 'models',
                'models': model_choices,
            }
            with open(os.path.join(history_dir, 'info.json'), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)

            with open(os.path.join(history_dir, 'results.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            DATA_STORE['results'] = results
            return redirect('results')
    else:
        form = UploadDatasetForm()
    return render(request, 'dataset_form.html', {
        'form': form,
        'title': 'Построение моделей',
        'button_text': 'Построить модели',
        'show_models': True
    })


def analyze_dataset(request):
    if request.method == 'POST':
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            df = pd.read_csv(csv_file)
            dataset_name = os.path.splitext(csv_file.name)[0]
            base_dir = os.path.join(settings.MEDIA_ROOT, 'analyze', dataset_name)
            os.makedirs(base_dir, exist_ok=True)

            plots = {}

            pairplot_path = os.path.join(base_dir, "pairplot.png")
            sns_plot = sns.pairplot(df.select_dtypes(include='number'))
            sns_plot.fig.savefig(pairplot_path)
            plt.close()
            plots['pairplot'] = f"analyze/{dataset_name}/pairplot.png"

            violins_dir = os.path.join(base_dir, 'violins')
            os.makedirs(violins_dir, exist_ok=True)
            violins = []

            numeric_cols = df.select_dtypes(include='number').columns
            for idx, col in enumerate(numeric_cols, start=1):
                fig, ax = plt.subplots(figsize=(4, 6))
                sns.violinplot(y=df[col], ax=ax)
                ax.set_title(f'{col}')
                path = os.path.join(violins_dir, f'{idx}.png')
                fig.savefig(path)
                plt.close(fig)
                violins.append(f"analyze/{dataset_name}/violins/{idx}.png")

            plots['violins'] = violins

            heatmap_path = os.path.join(base_dir, "heatmap.png")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            plt.tight_layout()
            fig.savefig(heatmap_path)
            plt.close(fig)
            plots['heatmap'] = f"analyze/{dataset_name}/heatmap.png"

            history_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
            history_dir = os.path.join(settings.BASE_DIR, 'history', history_id)
            os.makedirs(history_dir, exist_ok=True)

            info = {
                'dataset_name': dataset_name,
                'datetime': datetime.now().isoformat(),
                'method': 'analyze',
                'models': [],
            }
            with open(os.path.join(history_dir, 'info.json'), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)

            data_to_save = {
                'dtypes': df.dtypes.astype(str).to_dict(),
                'preview': df.head(10).to_dict(orient='list'),
                'plots': plots,
            }
            with open(os.path.join(history_dir, 'results.json'), 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)

            return render(request, 'analysis.html', {
                'dtypes': df.dtypes.to_dict(),
                'preview': df.head(10),
                'plots': plots,
                'media_url': settings.MEDIA_URL
            })
    else:
        form = UploadDatasetForm()
    return render(request, 'dataset_form.html', {
        'form': form,
        'title': 'Анализ датасета',
        'button_text': 'Анализировать',
        'show_models': False
    })


def results(request):
    results = DATA_STORE.get('results', {})
    return render(request, 'results.html', {
        'results': results,
        'media_url': settings.MEDIA_URL
    })


def history(request):
    history_root = os.path.join(settings.BASE_DIR, 'history')
    entries = []

    if os.path.exists(history_root):
        for dir_name in sorted(os.listdir(history_root), reverse=True):
            dir_path = os.path.join(history_root, dir_name)
            info_path = os.path.join(dir_path, 'info.json')
            if os.path.isdir(dir_path) and os.path.isfile(info_path):
                with open(info_path, encoding='utf-8') as f:
                    info = json.load(f)
                entries.append({
                    'id': dir_name,
                    'dataset_name': info.get('dataset_name'),
                    'datetime': info.get('datetime'),
                    'method': info.get('method'),
                    'models': info.get('models', [])
                })

    return render(request, 'history.html', {
        'entries': entries
    })


def history_detail(request, history_id):
    history_dir = os.path.join(settings.BASE_DIR, 'history', history_id)
    info_path = os.path.join(history_dir, 'info.json')
    results_path = os.path.join(history_dir, 'results.json')

    if not os.path.exists(info_path) or not os.path.exists(results_path):
        return render(request, '404.html', status=404)

    with open(info_path, encoding='utf-8') as f:
        info = json.load(f)
    with open(results_path, encoding='utf-8') as f:
        results = json.load(f)

    method = info.get('method')

    if method == 'models':
        return render(request, 'results.html', {
            'results': results,
            'media_url': settings.MEDIA_URL
        })

    elif method == 'analyze':
        dtypes = results.get('dtypes', {})
        preview_data = results.get('preview', {})
        plots = results.get('plots', {})

        # Преобразуем preview в строки
        columns = list(preview_data.keys())
        num_rows = len(next(iter(preview_data.values()))) if preview_data else 0
        rows = [
            [preview_data[col][i] for col in columns]
            for i in range(num_rows)
        ]

        return render(request, 'analysis.html', {
            'dtypes': dtypes,
            'preview': {
                'columns': columns,
                'values': rows
            },
            'plots': plots,
            'media_url': settings.MEDIA_URL
        })

    else:
        return render(request, '404.html', status=404)
