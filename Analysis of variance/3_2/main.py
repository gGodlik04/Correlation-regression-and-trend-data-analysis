import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Чтение данных из файла shops.csv
data = pd.read_csv('shops.csv')

# 1. Дисперсионный анализ без учета взаимодействия факторов
model = ols('price ~ C(store_type) + C(country)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 2. Дисперсионный анализ с учетом взаимодействия факторов
model_interaction = ols('price ~ C(store_type) * C(country)', data=data).fit()
anova_table_interaction = sm.stats.anova_lm(model_interaction, typ=2)
print(anova_table_interaction)

# 3. Проверка значимости фактора product с использованием теста Тьюки
tukey_result = pairwise_tukeyhsd(data['price'], data['product'])
print(tukey_result)

# Визуальное представление воздействия фактора product на цену
sns.barplot(x='product', y='price', data=data)