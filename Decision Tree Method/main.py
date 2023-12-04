import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Загрузка данных без заголовков
data = pd.read_csv('Bankruptcy.csv', header=None)

# Разделение данных на признаки и целевую переменную
X = data.iloc[:, :-1]  # все столбцы, кроме последнего
y = data.iloc[:, -1]  # последний столбец

# Преобразование категориальных признаков в числовые с помощью One-Hot кодирования
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 5])], remainder='passthrough')
X = ct.fit_transform(X)

# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели дерева решений
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")

# Вывод отчета о классификации
print(classification_report(y_test, y_pred))