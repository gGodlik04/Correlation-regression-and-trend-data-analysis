import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Загрузка данных из файла lda.txt
data = pd.read_csv('ida.csv', sep="\t")

# 1. Чтение данных и масштабирование
X = scale(data.iloc[:, :4])  # масштабирование первых 4 переменных-факторов
y = data.iloc[:, 4]  # классы

# 2. Создание обучающей и тестовой выборок
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 3. Проведение дискриминантного анализа
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 4. Интерпретация сводки модели
print("Априорные вероятности группы:")
print(lda.priors_)
print("Групповые средние:")
print(lda.means_)
print("Коэффициенты дискриминантных функций:")
print(lda.coef_)

# 5. Прогнозирование тестовых данных и оценка точности модели
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Точность предсказания моделью LDA:", accuracy)

# 6. Визуализация результатов
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

lda_scores = lda.transform(X_combined)
plt.scatter(lda_scores[:, 0], lda_scores[:, 1], c=y_combined, cmap='viridis')
plt.xlabel('ЛД1')
plt.ylabel('ЛД2')
plt.title('Дискриминантный анализ')
plt.show()

