import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
df = pd.read_csv('russian_passenger_air_service_2.csv', sep=',')

#Отбросим данные за 2020 год(из-за коронавируса)
df = df.drop(np.where(df['Year'] == 2020)[0])
print(df.head())

#Отбросим колонку с координатами аэропрота
df.drop(["Airport coordinates"], axis = 1, inplace = True)
print(df.head())

#Найдем средний пассажиропоток аэропортов по годам(кроме 2020)
data = df.groupby("Year").mean()
print(data)

mean_years = data["Whole year"].tolist()
print("Средние значения по годам", mean_years)

years = data.index.tolist()
print("Список годов = ", years)

plt.plot(years, mean_years)
plt.show()

#Построение гистограммы:
x1 = range(len(mean_years))
ax = plt.gca()
ax.bar(x1, mean_years, align='edge') # align='edge' - выравнивание по границе, а не по центру
ax.set_xticks(x1)
#ax.set_xticklabels(('first', 'second', 'third', 'fourth'))
plt.show()




