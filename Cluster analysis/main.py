import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('usa_elections.csv', sep=";")

# Удаление первого столбца (год), так как он не нужен для кластеризации
data = data.iloc[:, 1:]

# Замена значений NA на 0
data = data.fillna(0)

# Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Метод локтя для определения оптимального числа кластеров
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Построение графика для определения оптимального числа кластеров
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# На основе метода локтя выбираем число кластеров
n_clusters = 3

# KMeans кластеризация
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Интерпретация кластеров
cluster_centers = kmeans.cluster_centers_

# Вывод результатов
print("Результаты кластеризации:")
for i in range(n_clusters):
    print(f"Кластер {i + 1}:")
    states_in_cluster = data.index[clusters == i].tolist()
    print(states_in_cluster)
    print()
