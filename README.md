# ConvolutionalNeuralNetwork

Классификация изображений птиц (525 видов) свёрточной нейросетью с HTTP-интерфейсом.

Проект состоит из двух независимых частей:

| Часть | Язык / стек | Назначение | Состояние |
|---|---|---|---|
| `training/` | Python, PyTorch, torchvision | Дообучение ResNet-18 (transfer learning) на датасете птиц | Рабочий скрипт обучения, но веса никуда не сохраняются |
| `server/` | C++17, Boost.Asio, CMake, Docker | Многопоточный HTTP-сервер, который должен отдавать предсказание модели | Сервер работает, но классификатор — заглушка (код LibTorch закомментирован) |

---

## Структура репозитория

```
.
├── server/                      # C++ веб-сервис
│   ├── main.cpp                 # точка входа: io_service + HTTPServer на порту 8001
│   ├── WebServer.h / .cpp       # ThreadPool и HTTPServer
│   ├── ImageClassification.h/.cpp  # ImageClassifier (заготовка под LibTorch)
│   ├── CMakeLists.txt
│   ├── Dockerfile
│   ├── cmake-build-debug/       # артефакты сборки CLion (попали в git)
│   └── main, server             # собранные бинарники (попали в git)
└── training/                    # обучение модели
    ├── classification.py        # скрипт обучения ResNet-18
    ├── arch/                    # маленький поднабор: train/valid — 2 класса, test — 525 классов
    ├── archive/                 # полный датасет «BIRDS 525 SPECIES» (~2.2 ГБ)
    │   ├── train/ valid/ test/  # по 525 папок-классов
    │   ├── birds.csv            # метаданные: class id, filepaths, labels, data set, scientific name
    │   └── EfficientNetB0-525-(224 X 224)- 98.97.h5   # готовая Keras-модель из датасета (в коде не используется)
    └── hymenoptera_data/        # датасет ants/bees из туториала PyTorch (в коде не используется)
```

---

## Архитектура

### Общая схема (целевая)

```
 клиент (curl / браузер)
        │  GET /?<param>=<value>
        ▼
 ┌──────────────────────────── server ────────────────────────────┐
 │  boost::asio::io_service  ──►  HTTPServer::start_accept()      │
 │                                    │ async_accept              │
 │                                    ▼                           │
 │                         HTTPServer::handle_accept()            │
 │               занято ≥ 2 потоков? ──да──► 503 Service Unavailable
 │                                    │нет                        │
 │                                    ▼                           │
 │                    ThreadPool::enqueue(handle_request)         │
 │                                    ▼                           │
 │   handle_request: read_some → ImageClassifier::predict()       │
 │                   → getLogic() → HTTP/1.1 200 OK               │
 └────────────────────────────────────────────────────────────────┘
                                      ▲
                     веса модели (.pt) │  ← training/classification.py
```

Сейчас связь «обучение → сервер» не замкнута: скрипт обучения не сохраняет веса на диск, а сервер не подключает LibTorch.

### training/classification.py — обучение

Классический transfer learning по мотивам официального туториала PyTorch:

1. **Данные.** `datasets.ImageFolder` читает `data_dir/train` и `data_dir/valid` — одна подпапка на класс. Сейчас `data_dir = 'arch'` (маленький поднабор из 2 классов). Для полного датасета нужно указать `'archive'`.
2. **Предобработка.** `Resize(224, 224)` → `ToTensor()` → `Normalize` со статистиками ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Аугментаций нет, train и valid обрабатываются одинаково.
3. **Модель.** `torchvision.models.resnet18(weights='IMAGENET1K_V1')` — предобученный на ImageNet ResNet-18. Последний слой `fc` (512 → 1000) заменяется на `nn.Linear(512, 525)`.
4. **Обучение.**
   - функция потерь — `CrossEntropyLoss`;
   - оптимизатор — `SGD(lr=0.001, momentum=0.9)`, обновляется **только `fc`** (оптимизатору передаются `model_ft.fc.parameters()`). Градиенты по свёрточной части при этом всё равно считаются, потому что `requires_grad` у неё не отключён;
   - `StepLR(step_size=7, gamma=0.1)` — каждые 7 эпох lr делится на 10;
   - 25 эпох, `batch_size=4`, `num_workers=4`, устройство — `cuda:0`, если доступно, иначе CPU.
5. **Выбор лучшей модели.** После каждой эпохи считаются loss/accuracy на `valid`. Лучший по точности `state_dict` сохраняется во **временную** директорию (`TemporaryDirectory`) и в конце загружается обратно в модель. После выхода из функции временная директория удаляется — на диске весов не остаётся.

### server/ — веб-сервис на C++

**`main.cpp`** — создаёт `boost::asio::io_service`, `HTTPServer` на порту **8001** и запускает `service.run()`.

**`HTTPServer`** (`WebServer.cpp`):
- `start_accept()` — асинхронно принимает соединение (`async_accept`) и после каждого снова встаёт в ожидание.
- `handle_accept()` — если число активных задач в пуле ≥ размера пула (2), сразу отвечает `503 Service Unavailable`; иначе ставит `handle_request` в очередь пула.
- `handle_request()` — читает до 1024 байт запроса одним `read_some`, создаёт `ImageClassifier("path/to/model/weights", 525)`, вызывает `predict()` (сейчас возвращает пустую строку и результат не используется), формирует ответ через `getLogic()` и пишет его в сокет.
- `getLogic()` — «бизнес-логика»: берёт из сырой строки запроса всё после первого `=` и до первого пробела. По сути это **эхо значения query-параметра**: `GET /?img=bird.jpg HTTP/1.1` → тело ответа `bird.jpg`.

**`ThreadPool`** — самописный пул потоков: вектор `std::thread`, очередь `std::function<void()>`, `mutex` + `condition_variable`, счётчик `active_tasks` для отказа по перегрузке. Рабочий поток после взятия задачи вызывает `sleep(5)` — искусственная задержка 5 секунд для имитации тяжёлого инференса. Класс дополнительно наследуется от `boost::asio::thread_pool`, но его потоки не используются.

**`ImageClassifier`** (`ImageClassification.cpp`) — заготовка под LibTorch: в комментариях код загрузки ResNet-18, замены `fc`, `torch::load` весов, перевода на CUDA, чтения картинки через OpenCV, `softmax` + `argmax`. Сейчас всё закомментировано, `predict()` возвращает `""`.

---

## Запуск

### 1. Обучение модели

Требования: Python 3.9+, PyTorch и torchvision (GPU с CUDA желательно, но не обязательно), matplotlib, numpy, Pillow.

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy matplotlib pillow
python classification.py
```

Скрипт нужно запускать **из каталога `training/`**: путь к данным `arch` относительный.

Для обучения на полном датасете в `classification.py` поменяйте:

```python
data_dir = 'archive'
```

Чтобы веса остались после обучения (они понадобятся серверу), добавьте в конец скрипта:

```python
torch.save(model_ft.state_dict(), 'resnet18_birds.pt')
```

> Если нужен формат, который потом загрузит LibTorch в C++, проще всего сохранить TorchScript:
> `torch.jit.script(model_ft.cpu().eval()).save('resnet18_birds_ts.pt')` и загружать его в C++ через `torch::jit::load`.

### 2. Сервер — локальная сборка

Требования: g++ с поддержкой C++17, CMake ≥ 3.15, Boost (`system`, `thread`).

```bash
# Arch Linux
sudo pacman -S base-devel cmake boost
# Ubuntu / Debian
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev
```

Сборка через CMake:

```bash
cd server
cmake -S . -B build
cmake --build build
./build/server
```

Или напрямую через g++ (так делает Dockerfile):

```bash
cd server
g++ -std=c++17 -o main main.cpp WebServer.cpp ImageClassification.cpp -lboost_system -lboost_thread -lpthread
./main
```

Сервер слушает `0.0.0.0:8001`.

### 3. Сервер — Docker

```bash
cd server
docker build -t cnn-server .
docker run --rm -p 8001:8001 cnn-server
```

### 4. Проверка

```bash
curl "http://localhost:8001/?image=bird.jpg"
# через ~5 секунд: bird.jpg
```

Проверка отказа при перегрузке — пул на 2 потока, третий параллельный запрос получит 503:

```bash
for i in 1 2 3; do curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8001/?n=$i" & done; wait
```

---

## Известные проблемы и что доделать

**Обучение**
- Лучшие веса сохраняются только во временный каталог и пропадают после завершения — нужен `torch.save` в постоянный файл.
- `data_dir = 'arch'`: в `arch/train` и `arch/valid` всего 2 класса, а выходной слой на 525 — для такого набора это работает (метки 0 и 1), но фактически обучается бинарный классификатор. Для полной задачи — `archive`.
- Нет `if __name__ == '__main__':` — с `num_workers=4` на Windows/macOS (метод `spawn`) DataLoader упадёт.
- Нет аугментаций для train и нет оценки на `test`.
- Свёрточные слои не заморожены (`requires_grad=False`), поэтому backward идёт через всю сеть впустую.
- `archive/*.h5` (EfficientNetB0, Keras) и `hymenoptera_data/` в коде не используются.

**Сервер**
- `ImageClassifier` — заглушка: LibTorch не подключён ни в `CMakeLists.txt`, ни в Dockerfile, OpenCV не подключён вовсе.
- `ImageClassifier` создаётся заново на каждый запрос — при реальной модели её нужно загружать один раз при старте.
- `sleep(5)` вызывается **под захваченным мьютексом**: пока один поток «спит», второй не может взять задачу, а `get_active_tasks()` в `handle_accept` блокирует поток `io_service`. Задержку стоит вынести за пределы блокировки (или убрать).
- Размер пула захардкожен (`num_threads(2)` в конструкторе игнорирует аргумент), а наследование от `boost::asio::thread_pool` создаёт ещё 2 простаивающих потока.
- Ответ 503 отправляется синхронным `write` в потоке приёма; сокет после ответов явно не закрывается.
- HTTP разбирается вручную: только первые 1024 байта, без проверки метода/пути, без `Content-Type`. Загрузка самого изображения (multipart/POST) не реализована.
- В `Dockerfile` строка `RUN cmake . && cmake` лишняя, образ однослойный (сборочные инструменты остаются в итоговом образе).
- В git закоммичены `cmake-build-debug/`, `.idea/` и бинарники `main`/`server`. Они перечислены в `.gitignore`, но уже отслеживаемые файлы нужно один раз убрать из индекса: `git rm -r --cached server/cmake-build-debug server/.idea server/main server/server`.
