# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
#
# # Исходные данные
#
# salary=[]
# age=[]
# trust=[]
#
# data = np.loadtxt('2merF.txt')
# for line in data:
#     lineStr = str(line).replace("[", "")
#     words = lineStr.split()
#     age.append(float(words[1]))
#     salary.append(float(words[0]))
#     trust.append(float(words[2]))
# print(trust)
#
#
# data = pd.DataFrame(
#     {
#      'salary': salary,
#      'trust': trust,
#      'age': age
#     }
# )

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import shapiro, bartlett


# Чтение данных из файла
data = pd.read_csv('anova.csv', usecols=['type', 'sales'])
print(data)
# Перевод столбца type в факторный вид
data['type'] = pd.Categorical(data['type'])

# Построение Boxplot
sns.boxplot(x='type', y='sales', data=data)
sns.swarmplot(x='type', y='sales', data=data, color='black')

# Проверка нормальности распределения
shapiro_test = shapiro(data['sales'])
if shapiro_test[1] > 0.05:
    print('Распределение является нормальным')
else:
    print('Распределение не является нормальным')

# Проверка гомогенности дисперсий
bartlett_test = bartlett(*[data[data['type'] == t]['sales'] for t in data['type'].unique()])
if bartlett_test[1] > 0.05:
    print('Дисперсии являются однородными')
else:
    print('Дисперсии не являются однородными')

# Проведение дисперсионного анализа
model = smf.ols('sales ~ type', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Множественное сравнение с помощью теста Тьюки
tukeyhsd = sm.stats.multicomp.pairwise_tukeyhsd(data['sales'], data['type'])
print(tukeyhsd.summary())