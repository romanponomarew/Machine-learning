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



#МНК - Метод наименьших квадратов
#У нас есть множество точек(y = среднее количество пассажиров за год, х = год). Через них надо провести прямую, которая как можно ближе проходила к этим точкам.

#Проведем прямую y = kx + b через данные точки

x = np.array(years)
y = np.array(mean_years)

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

data1 = dict(zip(x_train, y_train))
print(data1)

#Перепишем линейное уравнение y = mx + c как y = Rp, где A = [[ x 1 ]] и p = [[m], [c]]
#Построим R по х :
R = np.vstack([x_train, np.ones(len(x_train))]).T
#Используем lstsq для решения его относительно вектора p
k, b = np.linalg.lstsq(R, y_train)[0]
#print(k, b)

def show_graf(x, y, k, b):
    #Построим график полученной прямой и укажем на нем точки
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, k*x + b, 'r', label='Fitted line')
    plt.legend()
    plt.show()
#show_graf(x, y, k, b)

#Рассчитаем кратчайшие расстояния от исходных данных(точек) до полученной прямой
h = abs(k*x_train-1*y_train+b)/((k**2 + (-1)**2)**0.5)
#Рассчитаем кратчайшие расстояния от исходных данных(точек) до полученной прямой
h = abs(k*x_train-1*y_train+b)/((k**2 + (-1)**2)**0.5)
print("Список полученных расстояний:", "\n", h)

#Функция для отображения гистограммы для передаваемого списка(h, y)
def show_hist(h):
    x1 = range(len(h))
    ax = plt.gca()
    ax.bar(x1, h, align='edge') # align='edge' - выравнивание по границе, а не по центру
    ax.set_xticks(x1)
    #ax.set_xticklabels(('first', 'second', 'third', 'fourth'))
    plt.show()

#Идея применения бутсрэпа в том, что у нас есть выборка небольшого размера и нам надо оценить, например, среднее.
# Вместо подсчета среднего самой этой выборки, мы извлекаем n_samples выборок с возвращением (то есть элементы могут повторяться) из исходной.
# У полученных выборок считаем среднее. Его уже оцениваем, вместо оценки среднего исходной выборки.
def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples
#Получим новую выборку из 1000 значений:
n_samples = 1000
number = 10
#Функция расчета доверительного интервала бутстреп методом(прогноз расстояния h):
def bootstrap_forecast(h, number, n_samples):

    spisok =[]
    for i in range(0, number):
        sample = get_bootstrap_samples(h, n_samples)
        spisok1 = spisok.append(np.mean(sample))
    #print("spisok =", spisok)
    global h_min
    global h_max
    h_min = min(spisok)
    h_max = max(spisok)
    print("Минимум доверительного интервала для расстояния =", h_min)
    print("Максимум доверительного интервала для расстояния =", h_max)

#bootstrap_forecast(h, number, n_samples)
#Прогноз будущих точек:
# Так, например, двигаясь по оси Ох, значение У будет принадлжать интервалу:
#h = abs(k*x-1*y+b)/((k**2 + (-1)**2)**0.5)
#Функция для расчета пассажиропотока на определенный месяц:

def forecast(x, h_min, h_max, y_pred):

    #Выше прямой МНК:
    y_max1 = -h_min*((k**2 + (-1)**2)**0.5) + k*x + b
    y_min1 = -h_max*((k**2 + (-1)**2)**0.5) + k*x + b
    #Ниже прямой МНК:
    y_min2 = h_min*((k**2 + (-1)**2)**0.5) - k*x - b
    y_max2 = h_max*((k**2 + (-1)**2)**0.5) - k*x - b
    #y = h*((k**2 + (-1)**2)**0.5) - k*x - b
    print(f"Максимальный прогнозируемый пассажиропоток для {x} года = ",y_max1)
    print(f"Минимальный прогнозируемый пассажиропоток для {x} года = ",y_min1)
    passengers = (y_max1 + y_min1)/2
    print(f"Прогнозируемый пассажиропоток за {x} год равен", passengers)
    y_pred.append(passengers)
    #print("y_min2 = ",y_min2)
    #print("y_max2 = ",y_max2)

#Отображение исходных данных и прямой МНК:
show_graf(x_train, y_train, k, b)
#Отображение гистограммы известного пассажиропотока:
show_hist(y_train)

#Отображение гистограммы известных расстояний точек до прямой:
#show_hist(h)

#Прогноз бутстрап методом, где h - исходные расстояния, number - количество генерируемых выборок
# для нахождения среднего значения выборки, n_samples - размер выборки:
bootstrap_forecast(h, number, n_samples)


y_pred = []
#Прогноз будущих цен для определенного месяца(вместо 2020 - прогнозируемый год, на выходе интервал возможных значений прогноза
for year in x_test:
    forecast(year, h_min, h_max, y_pred)


print(f'Прогнозируемый пассажиропоток за годы от {x_test[0]} до {x_test[len(x_test)-1]} равен', y_pred, sep='\n')
print(f"Реальный пассажиропоток за годы {x_test[0]} до {x_test[len(x_test)-1]} равен ", y_test)

# Сюжетные выходы
#Красным - прогнозируемый результат методом Бутстреп
plt.plot(x_test, y_pred, color='red', linewidth=3)
#Синим - настоящие результаты в тестовой выборке
plt.plot(x_test, y_test, color='blue', linewidth=3)
#Зеленым - тренировочная выборка
plt.plot(x_train, y_train, color='green', linewidth=3)
plt.show()