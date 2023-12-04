import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Исходные данные

salary=[]
trust=[]

data = np.loadtxt('1mer.txt')
for line in data:
	lineStr = str(line).replace("[","")
	words = lineStr.split()
	salary.append(float(words[0]))
	trust.append(float(words[1]))


data = pd.DataFrame(
    {
     'salary': salary,
     'trust': trust
    }
)

# Построение облака точек
plt.scatter(data['salary'], data['trust'])
plt.xlabel('Зарплата')
plt.ylabel('Платежеспособность')
plt.title('Облако точек: Платежеспособность от зарплаты')
plt.show()

# Максимизация функции правдоподобия
def log_likelihood(beta, x, y):
    # Логит-распределение
    z = beta[0] + beta[1]*x
    p = np.exp(z) / (1 + np.exp(z))
    # Функция правдоподобия
    ll = np.sum(y*np.log(p) + (1-y)*np.log(1-p))
    return -ll

x = data['salary']
y = data['trust']
initial_beta = [0, 0]  # Начальное значение коэффициентов

result = minimize(log_likelihood, initial_beta, args=(x, y))
beta = result.x  # Коэффициенты функции

# Логистическая кривая
def logit_function(x, beta):
    return np.exp(beta[0] + beta[1]*x) / (1 + np.exp(beta[0] + beta[1]*x))

# Добавление логистической кривой к облаку точек
plt.scatter(data['salary'], data['trust'])
x_vals = np.linspace(min(data['salary']), max(data['salary']), 100)  # 100 равномерно распределенных значений для построения кривой
plt.plot(x_vals, logit_function(x_vals, beta), color='red')  # Используем x_vals для построения логистической кривой
plt.xlabel('Зарплата')
plt.ylabel('Платежеспособность')
plt.title('Облако точек с логистической кривой')
plt.show()

# Проверка на других данных
test_data = pd.DataFrame({'salary': [10, 13, 15, 16, 21]})
test_data['probability'] = logit_function(test_data['salary'], beta)
test_data['approval'] = np.where(test_data['probability'] >= 0.5, 'одобрено', 'не одобрено')
print(test_data)