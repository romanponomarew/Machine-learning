import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
df = pd.read_csv('russian_passenger_air_service_2.csv', sep=',')
print("Верхние пять строчек исходного датасета: ")
print(df.head())
#Отбросим данные за 2020 год(из-за коронавируса)
df = df.drop(np.where(df['Year'] == 2020)[0])

#Отбросим колонку с координатами аэропрота
df.drop(["Airport coordinates"], axis = 1, inplace = True)

#Найдем средний пассажиропоток аэропортов по годам(кроме 2020)
data = df.groupby("Year").mean()
print("Измененный датасет под исследование:")
print(data)

mean_years = data["Whole year"].tolist()
print("Средние значения по годам", mean_years)

years = data.index.tolist()
print("Список годов = ", years)


plt.title("Средний пассажиропоток по годам(исходные данные)")
plt.xlabel("Год")
plt.ylabel("Пассажиропоток(человек)")
plt.plot(years, mean_years)
plt.show()

#Построение гистограммы:
#x1 = range(len(mean_years))
#x1 = range(years)

ax = plt.gca()
#ax.bar(x1, mean_years, align='edge') #align='edge' - выравнивание по границе, а не по центру
#ax.set_xticks(x1)
ax.bar(years, mean_years)
ax.set_title('Средний пассажиропоток по годам(исходные данные) - Гистограмма')
ax.set_xlabel('Год')
ax.set_ylabel('Пассажиропоток(человек)')

#ax.set_xticklabels(('first', 'second', 'third', 'fourth'))
plt.show()




