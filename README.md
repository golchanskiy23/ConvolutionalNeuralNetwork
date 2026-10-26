# ConvolutionalNeuralNetwork

Классификация изображений птиц (до 525 видов) свёрточной нейросетью с HTTP-интерфейсом.

Проект состоит из двух частей, связанных через файл модели:

| Часть | Язык / стек | Назначение |
|---|---|---|
| `training/` | Python, PyTorch, torchvision | Дообучение ResNet-18 (transfer learning) и экспорт модели в TorchScript |
| `server/` | C++20, LibTorch, Boost.Asio, CMake, Docker | Многопоточный HTTP-сервер: принимает изображение, возвращает класс птицы |

```
training/classification.py ──► server/model/model.pt + classes.txt ──► server (POST /predict)
```

---

## Структура репозитория

```
.
├── server/                      # C++ веб-сервис
│   ├── main.cpp                 # точка входа: загрузка модели, HTTPServer на порту 8001
│   ├── WebServer.h / .cpp       # ThreadPool и HTTPServer (разбор HTTP, маршрут /predict)
│   ├── ImageClassification.h/.cpp  # ImageClassifier: декодирование, предобработка, инференс LibTorch
│   ├── third_party/stb_image.h  # декодер JPEG/PNG/BMP/GIF (public domain, один заголовок)
│   ├── model/                   # сюда обучение кладёт model.pt и classes.txt (в .gitignore)
│   ├── CMakeLists.txt
│   └── Dockerfile
└── training/                    # обучение модели
    ├── classification.py        # обучение ResNet-18 + экспорт модели
    ├── arch/                    # маленький поднабор: train/valid — 2 класса, test — 525 классов
    ├── archive/                 # полный датасет «BIRDS 525 SPECIES» (~2.2 ГБ)
    │   ├── train/ valid/ test/  # по 525 папок-классов
    │   ├── birds.csv            # метаданные: class id, filepaths, labels, data set, scientific name
    │   └── EfficientNetB0-525-(224 X 224)- 98.97.h5   # готовая Keras-модель из датасета (в коде не используется)
    └── hymenoptera_data/        # датасет ants/bees из туториала PyTorch (в коде не используется)
```

---

## Архитектура

### Общая схема

```
 клиент (curl)
        │  POST /predict, тело = байты изображения
        ▼
 ┌──────────────────────────── server ─────────────────────────────┐
 │  main: ImageClassifier(model.pt, classes.txt) — один раз         │
 │                                                                  │
 │  поток io_context (весь сетевой ввод-вывод асинхронно):          │
 │    async_accept → Session                                        │
 │      async_read заголовков и тела (таймаут 15 с)                 │
 │      /predict? ──нет──► getLogic() → ответ сразу (/health, 404…) │
 │          │да                                                     │
 │      ThreadPool::try_enqueue ──все потоки заняты──► 503          │
 │          │                                                       │
 │  потоки пула (N, по умолчанию 2):                                │
 │      getLogic() → ImageClassifier::predict()                     │
 │      post(io_context) → async_write                              │
 │      200 {"class": "...", "confidence": 0.98}                    │
 └──────────────────────────────────────────────────────────────────┘
```

### training/classification.py — обучение

Классический transfer learning по мотивам официального туториала PyTorch:

1. **Данные.** `datasets.ImageFolder` читает `<data-dir>/train` и `<data-dir>/valid` — одна подпапка на класс. По умолчанию `--data-dir arch` (2 класса), для полного датасета — `--data-dir archive`.
2. **Предобработка.** `Resize(224, 224)` → `ToTensor()` → `Normalize` со статистиками ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Для train дополнительно аугментации: `RandomHorizontalFlip` и `ColorJitter`.
3. **Модель.** `torchvision.models.resnet18(weights='IMAGENET1K_V1')`. Все слои заморожены (`requires_grad=False`), последний `fc` заменяется на новый `nn.Linear(512, <число классов>)` — обучается только он.
4. **Обучение.** `CrossEntropyLoss`; `SGD(lr=0.001, momentum=0.9)`; `StepLR(step_size=7, gamma=0.1)`; `batch_size=4`; устройство — `cuda:0`, если доступно, иначе CPU.
5. **Выбор лучшей модели.** После каждой эпохи считается accuracy на `valid`, лучший `state_dict` загружается обратно в модель в конце обучения.
6. **Оценка на test.** Если есть `<data-dir>/test`, считается `Test Acc` — только по классам, на которых модель обучалась.
7. **Экспорт.** `export_model()` сохраняет в `--out-dir` (по умолчанию `../server/model`):
   - `model.pt` — модель в TorchScript (`torch.jit.script`), её загружает LibTorch без Python;
   - `classes.txt` — имена классов по строке; номер строки = индекс выхода модели.

### server/ — веб-сервис на C++

**`main.cpp`** — `./server [model.pt] [classes.txt] [port] [threads]` (по умолчанию `model/model.pt`, `model/classes.txt`, `8001`, `2`). Один раз загружает модель, запускает `HTTPServer` и корректно завершается по SIGINT/SIGTERM (Ctrl+C, `docker stop`).

**`ImageClassifier`** (`ImageClassification.cpp`):
- конструктор читает `classes.txt` и загружает модель через `torch::jit::load`;
- `predict(bytes)`: декодирует изображение через `stb_image` в RGB → тензор `1×3×H×W` в `[0, 1]` → билинейный resize до 224×224 со сглаживанием → нормализация ImageNet → `forward` в `torch::InferenceMode` → `softmax` → класс с максимальной вероятностью.
- Предобработка повторяет `data_transforms` из Python. Если менять её в обучении, нужно поменять и здесь. Вероятности сервера совпадают с Python с точностью ~0.005: PIL масштабирует `uint8`, сервер — `float`.

**`HTTPServer`** / **`Session`** (`WebServer.cpp`):
- `HTTPServer::start_accept()` — асинхронно принимает соединения и на каждое создаёт `Session`.
- `Session` — одно соединение. Весь сетевой ввод-вывод асинхронный и идёт в потоке `io_context`, поэтому медленный клиент не занимает поток инференса.
  - читает заголовки до `\r\n\r\n` (не больше 64 КБ, иначе `431`), затем тело по `Content-Length` (не больше 10 МБ, иначе `413`);
  - если запрос не пришёл целиком за 15 с, соединение закрывается;
  - `POST /predict` отправляет в пул потоков; если все потоки заняты — сразу `503`. Остальные запросы отвечаются без пула.
- `getLogic()` — маршрутизация: `POST /predict` → предсказание (`200`), `GET /health` → `{"status": "ok", "classes": N}`, неизвестный путь → `404`, другой метод → `405`, пустое тело или нераспознаваемое изображение → `400`, прочие ошибки → `500`. Все ответы — JSON с `Connection: close`.

**`ThreadPool`** — пул потоков для инференса: очередь `std::function<void()>`, `mutex` + `condition_variable`. `try_enqueue()` под одним мьютексом проверяет, что занято меньше потоков, чем есть в пуле, и только тогда ставит задачу. Поэтому очередь не растёт, а лишние запросы сразу получают 503. Модель одна на все потоки: вызывать `forward` TorchScript-модели из нескольких потоков одновременно безопасно.

---

## Запуск

### 1. Обучение и экспорт модели

Требования: Python 3.10–3.13 (под системный Python 3.14 колёс PyTorch может не быть — удобнее через `uv`), PyTorch, torchvision, numpy, Pillow, matplotlib. Без GPU NVIDIA ставьте CPU-сборку — она в несколько раз меньше.

```bash
cd training
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pillow matplotlib

python classification.py                      # arch, 25 эпох
python classification.py --epochs 2           # быстрая проверка (~3.5 мин на CPU)
python classification.py --data-dir archive   # полный датасет, 525 классов (на CPU — часы на эпоху)
```

Скрипт запускается **из каталога `training/`** (пути относительные). В конце печатается:

```
Model saved to ../server/model/model.pt, classes (2) to ../server/model/classes.txt
```

`torch.jit.script` в PyTorch 2.14 выдаёт `FutureWarning` о том, что TorchScript устарел. Пока он работает, это предупреждение можно игнорировать.

### 2. Сервер — Docker (рекомендуется)

Образ сам скачивает LibTorch 2.14.1 (CPU). Модель в образ не входит — она монтируется томом:

```bash
cd server
docker build -t cnn-server .
docker run --rm -p 8001:8001 -v "$PWD/model":/app/model:ro cnn-server
```

Версию LibTorch можно поменять: `docker build --build-arg LIBTORCH_VERSION=2.14.1 ...`. Лучше держать её равной версии PyTorch, которым сохранена модель.

### 3. Сервер — локальная сборка

Нужны: g++ с C++20, CMake ≥ 3.18, Boost (только заголовки Asio), LibTorch.

```bash
# Arch Linux
sudo pacman -S base-devel cmake boost
# Ubuntu / Debian
sudo apt install build-essential cmake libboost-dev
```

LibTorch можно взять из того же venv, где стоит PyTorch, либо скачать отдельно с https://pytorch.org (LibTorch, CPU):

```bash
cd server
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(../training/.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
# или -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
./build/server model/model.pt model/classes.txt 8001 2
```

При старте сервер печатает:

```
Loaded model model/model.pt (2 classes)
Listening on port 8001 with 2 inference threads, POST /predict
```

Число потоков инференса — четвёртый аргумент. В Docker: `docker run ... cnn-server ./server model/model.pt model/classes.txt 8001 4`.

### 4. Проверка

```bash
curl --data-binary "@training/arch/test/BLUE GROUSE/1.jpg" http://localhost:8001/predict
# {"class": "BLUE GROUSE", "confidence": 0.986391}
```

Состояние сервиса:

```bash
curl http://localhost:8001/health
# {"status": "ok", "classes": 2}
```

Ошибки:

```bash
curl http://localhost:8001/predict                                   # 405 Use POST ...
curl --data-binary "not an image" http://localhost:8001/predict      # 400 Cannot decode image
curl http://localhost:8001/foo                                       # 404
```

Перегрузка — пул на 2 потока, лишние параллельные запросы сразу получат 503:

```bash
for i in $(seq 10); do curl -s -o /dev/null -w "%{http_code}\n" \
  --data-binary "@training/arch/test/CACTUS WREN/1.jpg" http://localhost:8001/predict & done; wait
```

---

## Известные ограничения

**Обучение**
- По умолчанию обучение идёт на `arch` (2 класса), поэтому и сервер различает только 2 вида. Для 525 видов нужно `python classification.py --data-dir archive`. С замороженной свёрточной частью эпоха на `arch` идёт ~50 с на CPU; на `archive` (~85 тыс. изображений) — порядка часа на эпоху, с GPU — минуты.
- TorchScript помечен в PyTorch как устаревший (отсюда `FutureWarning`). Если его уберут, экспорт придётся перевести на `torch.export` + AOTInductor.
- `archive/*.h5` (EfficientNetB0, Keras) и `hymenoptera_data/` в коде не используются.

**Сервер**
- HTTP разбирается вручную: без keep-alive, chunked encoding и multipart. Изображение передаётся сырыми байтами в теле.
- Только CPU-инференс.
