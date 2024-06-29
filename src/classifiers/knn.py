import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import joblib

MODEL_NAME = "KNN"
ATTACK_NAME = "fault"
DATASET_NAME = "mapped_dataset.csv"
K_NEIGHBORS = 10

plots_dir = f'{ATTACK_NAME}/{MODEL_NAME}/{DATASET_NAME}/{K_NEIGHBORS}/plots'
results_dir = f'{ATTACK_NAME}/{MODEL_NAME}/{DATASET_NAME}/{K_NEIGHBORS}/results'
models_dir = f'{ATTACK_NAME}/{MODEL_NAME}/{DATASET_NAME}/{K_NEIGHBORS}/models'

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Carregar o conjunto de dados (substitua 'seu_arquivo.csv' pelo caminho do seu arquivo CSV real)
data = pd.read_csv(f'dataset/{ATTACK_NAME}/{DATASET_NAME}')

# Dividir o conjunto de dados em features (X) e rótulos (y)
X = data.drop(f'{ATTACK_NAME}', axis=1)  # Substitua 'target_column' pelo nome da coluna de destino
y = data[f'{ATTACK_NAME}']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o classificador KNN
model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)

start_time_training = time.time()

model.fit(X_train, y_train)

end_time_training = time.time()

training_time = end_time_training - start_time_training
print(f"Tempo de treinamento: {training_time:.2f} segundos")

start_time_testing = time.time()

y_pred = model.predict(X_test)

end_time_testing = time.time()

testing_time = end_time_training - start_time_testing
print(f"Tempo de teste: {testing_time:.2f} segundos")

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calcular a acurácia do modelo no conjunto de teste
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia no teste:", accuracy)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig(f'{plots_dir}/confusion_matrix.png')  # Salvar o gráfico
plt.close()

# Calcular e exibir precision, recall e f1-score
report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report:")
print(report)

# Calcular a acurácia no conjunto de treinamento
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print("Acurácia no treinamento:", train_accuracy)

# Avaliar o modelo usando validação cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Gerar learning curves
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calcular média e desvio padrão das pontuações de treinamento e teste
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plotar learning curves
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.title('Learning Curves')
plt.savefig(f'{plots_dir}/learning_curves.png')  # Salvar o gráfico
plt.close()
"""
# Plotar ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(f'{plots_dir}/roc_curve.png')
plt.close()

# Plotar Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(f'{plots_dir}/precision_recall_curve.png')
plt.close()

# Plotar histograma das previsões
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_proba, bins=10, kde=True, color='blue')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')
plt.savefig(f'{plots_dir}/predicted_probabilities_histogram.png')
plt.close()
"""
# Salvar os resultados em um arquivo JSON
results = {
    "test_accuracy": accuracy,
    "classification_report": report,
    "train_accuracy": train_accuracy,
    "cross_validation_scores": cv_scores.tolist(),
    "mean_cross_validation_score": cv_scores.mean(),
    "training_time": training_time
}

with open(f'{results_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Salvar o modelo treinado
joblib.dump(model, f'{models_dir}/model.joblib')
