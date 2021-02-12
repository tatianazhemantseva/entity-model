SkillFactory

Дипломный проект Жеманцевой Татьяны

Никнейм на Kaggle tatianazhemantseva

Цель:
построение модели, которая по представленной картинке кожного покрова сможет определить варианты с процентным соотношением похожести данного дефекта на известное заболевание, например, "данный дефект похож с вероятностью 60% на фиброму, с вероятностью 15% на кератоз и с вероятностью 5% на родинку".
На просторах интернета пользователи довольно часто задаются вопросом "ЧТО ЭТО?" при появлении каких-то дефектов на коже. В частности на эту задачу меня сподвигла идея создания приложения для группы в FB "Аллергомамы" с количеством участников 16482, в которой ежедневно появляется минимум 3 вопроса "на что это похоже?" с приложенными фото кожи детей. 

Данные:
Данные взяты с закрытых соревнований. Для соревнования данные предоставлены и отсортированы в ISIC (https://www.isic-archive.com) - Институт специализирующийся на заболеваниях кожи. Представляют собой изображения различных кожных дефектов. В работе использовались датасеты с 23 классами и 9 классами. Основной - с 23 классами.
https://www.kaggle.com/nodoubttome/skin-cancer9-classesisic

Данный набор состоит из 15550 изображений злокачественных и доброкачественных кожных заболеваний, которые были сформированы при Международном сотрудничестве в области визуализации кожи (ISIC). Все изображения были отсортированны в соответстие с классификацией взятой с ISIC, и все поднаборы были разделены на группы изображений.
Набор данных содержит в себе данные по дефектам кожи, например, старческий кератоз, фиброма кожи, меланома, родинка и т.д.

В ходе работы над проектом были решены следующие задачи:

→ EDA

Проведен анализ по распределению изображений по классам.
Применены дополнительные методы предобработки изображений (аугментация с помощью albumentations и ImageDataGenerator).

→ Построение модели по обработке естественного языка (NLP)

Обучение модели: 
1. Использование различных LR, optimizer(adam, sgd), activation(softmax, relu). 
2. Использование других архитектур и/или их ансамблей (SOTA-решения): Xception, EfficientNetB4, InceptionV3, ResNet50, VGG19. Применение Fine-tuning c переносом обучения. 
3. Применение функций callback в Keras: ReduceLROnPlateau, ModelCheckpoint, EarlyStopping.
4. softmax используется последней функцией активации нейронной сети , чтобы нормализовать выход сети к распределению вероятностей по предсказанным выходным классам.

→ Оценка модели и интерпретация результатов
На разных архитектурах получен предел модели 60%. f1-score= 55%. Разброс f1-score по классам от 47 до 82.
Модель с большим набором данных для обучения в confusion matrix содержит большинство предсказаний по диагонали, что говорит об относительно точном предсказании модели по классу.
Для улучшения предсказания использовалось усреднение по нескольким предсказаниям с изменением исходного изображения, использование весов классов, разные аугментации, Fune Tuning, подбор LR, уменьшение батча, увелечение количества эпох.

Файлы в репозитории Репозиторий содержит два файлы:
- основной ноутбук с разработкой модели

Значение метрики, которого удалось добиться 60%

Что не получилось сделать так, как хотелось? Над чем ещё стоит поработать?
Точность модели конечно разочаровала, попытки увеличить ее известными способами привели только к потере времени.
Чтобы хотелось сделать еще - решить эту задачу с помощью сегментации и классификации (Self-Supervised Image Classification).
