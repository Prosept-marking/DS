# DS

### Цель проекта: разработка решения, которое автоматизирует процесс сопоставления товаров заказчика с размещаемыми товарами дилеров.

✏️**Распределение задач:**

**Евгений:** предобработка данных, разработка решения методом ближайших соседей с классификатором KNN, создание скрипта\
**Дарья:** предобработка данных, разработка решения методом ближайших соседей с CatBoost\
**Михаил:** разработка решения путем вычисления косинусного сходства между векторами названий товаров

:book: **Описание данных:**
>*marketing_dealerprice:* парсинг сайтов дилеров\
*marketing_product:* продукты компании Prosept\
*marketing_productdealerkey:* матчинг товаров заказчика и дилеров

**В процессе работы над проектом:**
1. Были исследованы предоставленные данные, выбраны подходы к решению задачи.
2. Проведена предобработка данных. Созданы функции очистки, лематизации и эмбеддинга названий продуктов базы Prosept и запросов дилеров.
3. Реализованы 3 подхода к решению задачи автоматизации сопоставления товаров.
4. Выбран лучший из них и разработана модель KNN
5. Оформлен скрипт

**Метрика качества:** precision@5

**Инструкция:**

