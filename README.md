# Проект по созданию ИИ-сервиса
Вот ссылка на OneNote: https://onedrive.live.com/personal/9651c36a49030668/_layouts/15/Doc.aspx?sourcedoc=%7B46a6af1c-7ffc-41d2-b9d5-636dfa3e59b6%7D&action=edit&redeem=aHR0cHM6Ly8xZHJ2Lm1zL28vYy85NjUxYzM2YTQ5MDMwNjY4L0lnQWNyNlpHX0hfU1FiblZZMjM2UGxtMkFXdWFsQUpXTE5jbkRFQ2kyVDFINnJr&wd=target%28Стандарты%20разработки.one%7C8a819970-097c-3748-a6d1-591cc671d95d%2FСтраница%20без%20заголовка%7Cffe4ca1b-d191-a849-ae74-71eed99c5a02%2F%29&wdorigin=NavigationUrl \
Там тетрадка, куда я буду записывать свои наблюдения по ходу разработки


## Описание
Исследование направлено на область медицины - патологию. Результаты (надеюсь) помогут в определении опухолей на ранних стадиях, предотвращения возможных осложнений или летальных исходов\
\
Модели, используемые в проекте:\
U-Net\
![Изображение](https://medicine.utah.edu/sites/g/files/zrelqx421/files/migration/media/unet-graphic.png)



### Структура проекта (будет меняется)
**src** - коренная папка проекта (не считая всяких README или подобных файлов)

#### Папки
**api** - back-end часть разработки, где находятся всякие router -ы, для API (интерфейс программирования приложения)\
**models** - папка, в которой лежат всякие databases\
**nn** - модель(-и) машинного обучения\
**schemas** - pydantic классы

#### Другие файлы
**main.py** - код, через который запускается приложение\
**config** - код для взаимодействия с настройками базы данных\
**.env** - настройки базы данных postgres
