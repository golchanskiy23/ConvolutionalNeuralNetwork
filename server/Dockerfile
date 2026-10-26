FROM ubuntu:24.04 AS build

# Установите необходимые зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libtool \
    pkg-config  \
    g++ \
    cmake \
    wget \
    libboost-dev \
    libboost-thread-dev \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

# Установите рабочую директорию
WORKDIR /src

# Скопируйте файлы проекта в контейнер
COPY CMakeLists.txt main.cpp WebServer.cpp WebServer.h ImageClassification.cpp ImageClassification.h ./

# Установите переменную окружения для CMAKE_PREFIX_PATH
#ENV CMAKE_PREFIX_PATH=/usr/local/libtorch

# Создайте директорию для сборки
RUN g++ -std=c++17 -o main main.cpp WebServer.cpp ImageClassification.cpp -lboost_thread -lpthread

#RUN chmod +x server

EXPOSE 8001

# Укажите команду по умолчанию для запуска приложения
CMD ["./main"]

#FROM ubuntu:latest

#WORKDIR /opt/server

#COPY --from=build /src/server_demo ./
