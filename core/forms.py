from django import forms

MODEL_CHOICES = [
    ('linear', 'Линейная регрессия'),
    ('tree', 'Дерево решений'),
    ('knn', 'K-ближайших соседей'),
    ('svr', 'Support Vector Regression'),
    ('lasso', 'Лассо-регрессия'),
    ('rf', 'Random Forest (ансамблевая)'),
    ('boost', 'Gradient Boosting (ансамблевая)'),
    ('bagging', 'Bagging Regressor (ансамблевая)'),
]

class UploadDatasetForm(forms.Form):
    csv_file = forms.FileField(label='Выберите CSV файл')
    models = forms.MultipleChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label='Выберите модели',
        required=False
    )
