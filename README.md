SkillFactory

Дипломный проект Жеманцевой Татьяны

Никнейм на Kaggle tatianazhemantseva

Цель:
построение модели, которая по представленной картинке кожного покрова сможет определить варианты с процентным соотношением похожести данного дефекта на известное заболевание, например, "данный дефект похож с вероятностью 60% на фиброму, с вероятностью 15% на кератоз и с вероятностью 5% на родинку".
На просторах интернета пользователи довольно часто задаются вопросом "ЧТО ЭТО?" при появлении каких-то дефектов на коже. В частности на эту задачу меня сподвигла идея создания приложения для группы в FB "Аллергомамы" с количеством участников 16482, в которой ежедневно появляется минимум 3 вопроса "на что это похоже?" с приложенными фото кожи детей. 

Данные:
Данные взяты с закрытого соревнования. Представляют собой картинки различных кожных дефектов. Всего 23 класса.  
https://www.kaggle.com/nodoubttome/skin-cancer9-classesisic

Данный набор состоит из 2357 изображений злокачественных и доброкачественных кожных заболеваний, которые были сформированы при Международном сотрудничестве в области визуализации кожи (ISIC). Все изображения были отсортированны в соответстие с классификацией взятой с ISIC, и все поднаборы были разделены на одинаковое количество изображений, за исключением меланом и родинок, чьи изображения немного преобладают.
Набор данных содержит в себе данные по следующим заболеваниям:

старческий кератоз
базально-клеточная карцинома
фиброма кожи
меланома
родинка
пигментный доброкачественный кератоз
себорейный кератоз
плоскоклеточный рак
сосудистое поражение

В ходе работы над проектом были решены следующие задачи:

→ EDA

Проведен анализ по распределению изображений по классам.
Применены дополнительные методы предобработки изображений (аугментация с помощью albumentations и ImageDataGenerator).

→ Построение модели по обработке естественного языка (NLP)

Обучение модели: Использование различных LR, optimizer(adam, sgd), activation(softmax, relu, sigmoid). Использование разной аугментации с разными библиотеками. Применение Fine-tuning c переносом обучения. Использование других архитектур и/или их ансамблей (SOTA-решения): Xception, EfficientNetB4, InceptionV3, ResNet50, VGG19. Применение функций callback в Keras.
softmax используется последней функцией активации нейронной сети , чтобы нормализовать выход сети к распределению вероятностей по предсказанным выходным классам.

→ Оценка модели и интерпретация результатов



Файлы в репозитории Репозиторий содержит два файлы:
- основной ноутбук с разработкой модели

Значение метрики, которого удалось добиться 



Что не получилось сделать так, как хотелось? Над чем ещё стоит поработать?

в этом проекте не удалось лично дойти до практики с методами обучения 
