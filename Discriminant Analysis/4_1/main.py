import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import bartlett, shapiro, norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Загрузка данных из файлов
df = pd.read_csv('dataset_1.csv', sep=" ", header=None)

df1 = df[[0]]
df2 = df[[1]]


# Проверка нормальности распределения фактора для каждого класса
_, p_value1 = shapiro(df1)
_, p_value2 = shapiro(df2)

# Построение гистограмм распределения фактора для каждого класса
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df1, bins='auto', alpha=0.7, color='blue')
plt.xlabel('x1')
plt.ylabel('Frequency')
plt.title('Class 1')

plt.subplot(1, 2, 2)
plt.hist(df2, bins='auto', alpha=0.7, color='orange')
plt.xlabel('x2')
plt.ylabel('Frequency')
plt.title('Class 2')

plt.tight_layout()
plt.show()

# Нахождение априорных вероятностей для каждого класса
n1 = len(df1)
n2 = len(df2)
total = n1 + n2
prior_prob1 = n1 / total
prior_prob2 = n2 / total

# Нахождение оценок параметров: средние значения и среднеквадратические отклонения для каждого класса
mean1 = np.mean(df1)
mean2 = np.mean(df2)
std1 = np.std(df1)
std2 = np.std(df2)
pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)

# Проверка гипотезы о равенстве дисперсий
_, p_value = scipy.stats.bartlett(df1.to_numpy().flatten(), df2.to_numpy().flatten())
is_equal_variance = p_value > 0.05


# Функция для прогнозирования принадлежности к классам
def discriminant_function(x):
    g1 = norm.pdf(x, mean1, pooled_std) + np.log(prior_prob1)
    g2 = norm.pdf(x, mean2, pooled_std) + np.log(prior_prob2)

    if g1 > g2:
        return 1
    else:
        return 2


# Прогноз для новых значений фактора
new_x = np.array([5.0])  # Преобразуйте входные данные в массив
prediction = discriminant_function(new_x)
print(f"The new value {new_x} belongs to class {prediction}.")