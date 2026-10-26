FROM ubuntu:24.04 AS build

ARG LIBTORCH_VERSION=2.14.1

# Установите необходимые зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    ca-certificates \
    libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# LibTorch (CPU). Версия должна совпадать с PyTorch, которым сохранена модель
RUN wget -q -O /tmp/libtorch.zip \
    "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip" \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

# Установите рабочую директорию
WORKDIR /src

# Скопируйте файлы проекта в контейнер
COPY CMakeLists.txt main.cpp WebServer.cpp WebServer.h ImageClassification.cpp ImageClassification.h ./
COPY third_party ./third_party

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch \
    && cmake --build build -j"$(nproc)"

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Библиотеки LibTorch по тому же пути, что и при сборке (RPATH)
COPY --from=build /opt/libtorch/lib /opt/libtorch/lib
COPY --from=build /src/build/server /app/server

WORKDIR /app

EXPOSE 8001

# Модель монтируется в /app/model: model.pt + classes.txt
CMD ["./server", "model/model.pt", "model/classes.txt", "8001"]
