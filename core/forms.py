from django import forms

MODEL_CHOICES = [
    ('linear', 'Линейная регрессия'),
    ('tree', 'Дерево решений'),
    ('rf', 'Random Forest (ансамблевый)'),
    ('boost', 'Gradient Boosting (ансамблевый)'),
]

class UploadDatasetForm(forms.Form):
    csv_file = forms.FileField(label='Выберите CSV файл')
    models = forms.MultipleChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label='Выберите модели',
    )