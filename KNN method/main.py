import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Чтение данных
data = pd.read_csv('Bankruptcy.csv')

# Преобразование категориальных переменных в числовые
label_encoder = LabelEncoder()
for column in data.columns[:-1]:
    data[column] = label_encoder.fit_transform(data[column])

# Разделение данных на факторы (X) и целевую переменную (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Создание и обучение модели KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Предсказание классов для тестовых данных
y_pred = knn.predict(X_test)

print(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность предсказания моделью KNN:", accuracy)