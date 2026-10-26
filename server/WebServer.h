#ifndef IMAGE_CLASSIFICATION_WEBSERVER_H
#define IMAGE_CLASSIFICATION_WEBSERVER_H

#include <boost/asio.hpp>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "ImageClassification.h"
using namespace boost::asio;

// Пул потоков для инференса. Принимает задачу, только если есть свободный поток,
// иначе сервер сразу отвечает 503 и не копит очередь.
class ThreadPool {
  public:
   explicit ThreadPool(size_t num_threads);

   ~ThreadPool();

   // Ставит задачу в работу, если занято меньше потоков, чем есть в пуле
   bool try_enqueue(std::function<void()> task);

   size_t size() const { return threads_.size(); }

  private:
   std::vector<std::thread> threads_;
   std::queue<std::function<void()>> tasks_;
   std::mutex mutex_;
   std::condition_variable condition_;
   bool stop_ = false;
   size_t busy_ = 0;  // задачи в очереди + выполняющиеся
};

class HTTPServer {
  public:
   HTTPServer(io_context& io_context, unsigned short port, ImageClassifier& classifier, size_t num_threads);

  private:
   friend class Session;

   // business logic function
   std::string getLogic(const std::string& method, const std::string& target, const std::string& body);
   void start_accept();

   ip::tcp::acceptor acceptor_;
   io_context& io_context_;
   ImageClassifier& classifier_;
   ThreadPool pool;
};

// Одно HTTP-соединение: асинхронное чтение запроса с таймаутом,
// инференс в пуле потоков, асинхронная запись ответа.
class Session : public std::enable_shared_from_this<Session> {
  public:
   Session(ip::tcp::socket socket, HTTPServer& server);

   void start();

  private:
   void read_headers();
   void read_body(size_t content_length);
   void process();
   void write_response(std::string response);

   ip::tcp::socket socket_;
   HTTPServer& server_;
   steady_timer deadline_;
   streambuf buf_;
   std::string method_, target_, body_, response_;
};

#endif //IMAGE_CLASSIFICATION_WEBSERVER_H
