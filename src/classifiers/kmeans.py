import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import mode

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = Path(SRC_PATH).parent.parent
DATASETS_PATH = os.path.join(ROOT_PATH, 'dataset')
OUTPUTS_PATH = os.path.join(ROOT_PATH, 'outputs')
PLOTS_PATH = os.path.join(ROOT_PATH, 'plots')

MODEL_NAME = "KMeans"
ATTACK_NAME = "fault"
DATASET_NAME = "mapped_dataset_2.csv"
FILE_PATH = os.path.join(DATASETS_PATH, ATTACK_NAME, DATASET_NAME)

data = pd.read_csv(FILE_PATH)

plots_dir = f'../../{ATTACK_NAME}/{MODEL_NAME}/{DATASET_NAME}/plots'
results_dir = f'../../{ATTACK_NAME}/{MODEL_NAME}/{DATASET_NAME}/results'
models_dir = f'../../{ATTACK_NAME}/{MODEL_NAME}/{DATASET_NAME}/models'

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

X = data.drop(f'{ATTACK_NAME}', axis=1)
y = data[f'{ATTACK_NAME}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ajustar o modelo KMeans
n_clusters = len(np.unique(y))  # Número de clusters igual ao número de classes
model = KMeans(n_clusters=n_clusters, random_state=42)

start_time_training = time.time()
model.fit(X_train)
end_time_training = time.time()

training_time = end_time_training - start_time_training
print(f"Tempo de treinamento: {training_time:.2f} segundos")

# Prever clusters para o conjunto de teste
y_pred_clusters = model.predict(X_test)

# Mapeamento dos clusters para os rótulos verdadeiros
labels = np.zeros_like(y_pred_clusters)
for i in range(n_clusters):
    mask = (y_pred_clusters == i)
    labels[mask] = mode(y_test[mask])[0]

# Calcular acurácia
accuracy = accuracy_score(y_test, labels)
print("Acurácia no teste:", accuracy)

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig(f'{plots_dir}/confusion_matrix.png')
plt.close()

# Relatório de classificação
report = classification_report(y_test, labels, output_dict=True)
print("Classification Report:")
print(report)

# Usar PCA para reduzir a dimensionalidade dos dados para 2D para visualização
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Plotar os clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_clusters, cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title('KMeans Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig(f'{plots_dir}/kmeans_clustering.png')
plt.close()

# Plotar learning curves usando dummy scores (não aplicável diretamente ao KMeans, mas ilustrativo)
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.title('Learning Curves')
plt.savefig(f'{plots_dir}/learning_curves.png')
plt.show()
plt.close()

# Plotar histograma das previsões de clusters
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_clusters, bins=n_clusters, kde=True, color='blue')
plt.xlabel('Predicted Clusters')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Clusters')
plt.savefig(f'{plots_dir}/predicted_clusters_histogram.png')
plt.close()

results = {
    "test_accuracy": accuracy,
    "classification_report": report,
    "training_time_seconds": training_time
}

with open(f'{results_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

joblib.dump(model, f'{models_dir}/model.joblib')
