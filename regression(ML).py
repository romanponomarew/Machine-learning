import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
#mean_years = data["Whole year"]
print("Средние значения по годам", mean_years)

#years = data.index.tolist()
years = data.index.tolist()
print("Список годов = ", years)


# регрессия с использованием набора данных
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

y = np.array(mean_years)
x = np.array(years).reshape((-1, 1))


# Разделяем данные на обучающие / тестовые наборы
#Первая половина доступной выборки - обучающая
x_train = x[:(len(x)//2)]
#Вторая половина доступной выборки - тестовая, для проверки
x_test = x[(len(x)//2):]
print("x_train=", x_train)
print("x_test=", x_test)

#Такой же принцип и с Y - разделение на обучающие и тестовые наборы
y_train = y[:(len(y)//2)]
y_test = y[(len(y)//2):]
#print("y_train=", y_train)
#print("y_test=", y_test)

# Сюжетные выходы
plt.scatter(x_test, y_test,  color='black')

plt.title('Test Data')
plt.xlabel('Size')

plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

#Создаем экземпляр класса LinearRegression, это будет моделью линейной регрессии
model = linear_model.LinearRegression()

# Тренируем модель, используя тренировочные наборы(автоматический подбор параметров модели)
model.fit(x_train, y_train)

# Сюжетные выходы
#Красным - прогнозируемый результат методом линейной регрессии
plt.plot(x_test, model.predict(x_test), color='red', linewidth=3)
#Синим - настоящие результаты в тестовой выборке
plt.plot(x_test, y_test, color='blue', linewidth=3)
#Зеленым - тренировочная выборка
plt.plot(x_train, y_train, color='green', linewidth=3)
plt.show()
#На этом графике, мы наносим тестовые данные. Красная линия указывает линию наилучшего соответствия для прогнозирования Y.

#Индивидуальный прогноз с использованием модели линейной регрессии:

y_pred = model.predict(x_test)
print(f'Прогнозируемый пассажиропоток за годы от {x_test[0]} до {x_test[len(x_test)-1]} равен', y_pred, sep='\n')
print(f"Реальный пассажиропоток за годы {x_test[0]} до {x_test[len(x_test)-1]} равен ", y_test)