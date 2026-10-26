#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind/bind.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <algorithm>
#include "WebServer.h"
#include "ImageClassification.h"

using namespace boost::asio;
typedef boost::shared_ptr<ip::tcp::socket> socket_ptr;

   ThreadPool::ThreadPool(size_t num_threads) : boost::asio::thread_pool(num_threads) , num_threads(2), active_tasks(0)
   {
     for (size_t i = 0; i < num_threads; ++i) {
       threads_.emplace_back([this] {
         while (true) {
           std::function<void()> task;
           {
             std::unique_lock<std::mutex> lock(mutex_);
             condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
             if (stop_ && tasks_.empty()) {
               return;
             }
             task = std::move(tasks_.front());
             tasks_.pop();
             active_tasks++;
           }
           task();
           {
             std::lock_guard<std::mutex> lock(mutex_);
             --active_tasks;
             condition_.notify_all();
           }

         }
       });
     }
   }

   void ThreadPool::sleep(int duration)
   {
     std::this_thread::sleep_for(std::chrono::seconds (duration));
   }

   template<class F>
   void ThreadPool::enqueue(F&& f) {
     {
       std::unique_lock<std::mutex> lock(mutex_);
       tasks_.emplace(std::forward<F>(f));
     }
     condition_.notify_one();
   }

   ThreadPool::~ThreadPool() {
     {
       std::unique_lock<std::mutex> lock(mutex_);
       stop_ = true;
     }
     condition_.notify_all();
     for (auto& thread : threads_) {
       thread.join();
     }
   }

   int ThreadPool::getQueueSize()
   {
     return this->tasks_.size();
   }

   int ThreadPool::getSize()
   {
     return this->num_threads;
   }

   size_t ThreadPool::get_active_tasks()
   {
     std::lock_guard<std::mutex> lock(mutex_);
     return active_tasks;
   }

   // Максимальный размер тела запроса (изображения)
   static const size_t kMaxBodySize = 10 * 1024 * 1024;

   static std::string makeResponse(const std::string& status, const std::string& body,
                                   const std::string& contentType = "application/json")
   {
     return "HTTP/1.1 " + status + "\r\n"
            "Content-Type: " + contentType + "\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n\r\n" + body;
   }

   static std::string jsonEscape(const std::string& str)
   {
     std::string result;
     for (char c : str) {
       if (c == '"' || c == '\\') result += '\\';
       if (static_cast<unsigned char>(c) < 0x20) continue;
       result += c;
     }
     return result;
   }

   static std::string jsonError(const std::string& message)
   {
     return "{\"error\": \"" + jsonEscape(message) + "\"}\n";
   }

   HTTPServer::HTTPServer(io_context& io_context, short port, ImageClassifier& classifier)
     : acceptor_(io_context, ip::tcp::endpoint(ip::tcp::v4(), port)), io_context_(io_context),
       classifier_(classifier), pool(2)
   {
     start_accept();
   }

   // business logic function
   std::string HTTPServer::getLogic(const std::string& method, const std::string& target, const std::string& body)
   {
     if (target != "/predict") {
       return makeResponse("404 Not Found", jsonError("Unknown path, use POST /predict"));
     }
     if (method != "POST") {
       return makeResponse("405 Method Not Allowed", jsonError("Use POST with image bytes in body"));
     }
     if (body.empty()) {
       return makeResponse("400 Bad Request", jsonError("Empty body, send image bytes"));
     }
     try {
       Prediction prediction = classifier_.predict(body);
       return makeResponse("200 OK", "{\"class\": \"" + jsonEscape(prediction.className) +
                                     "\", \"confidence\": " + std::to_string(prediction.confidence) + "}\n");
     } catch (std::invalid_argument& e) {
       return makeResponse("400 Bad Request", jsonError(e.what()));
     }
   }

   void HTTPServer::start_accept() {
     socket_ptr socket(new ip::tcp::socket(io_context_));
     acceptor_.async_accept(*socket,
                            [this, socket](auto && PH1) { handle_accept(socket, std::forward<decltype(PH1)>(PH1)); });
   }

   void HTTPServer::handle_accept(const socket_ptr socket, const boost::system::error_code& error) {
     if (!error) {
       if(this->pool.get_active_tasks() >= this->pool.getSize()) {
         //write(*socket,buffer("HTTP/1.1 400 Bad Request\r\n\r\n"));
         write(*socket, boost::asio::buffer(makeResponse("503 Service Unavailable", jsonError("All threads are busy"))));
         std::cerr << "Error: All threads are busy. Returning 503 Service Unavailable.\n";
       }
       else pool.enqueue([this, socket] { handle_request(socket); });
     }
     start_accept();
   }

   void HTTPServer::handle_request(const socket_ptr socket) {
     try {
       // Читаем заголовки целиком
       streambuf request_buf;
       size_t header_length = read_until(*socket, request_buf, "\r\n\r\n");
       std::string data(buffers_begin(request_buf.data()), buffers_end(request_buf.data()));
       std::string headers = data.substr(0, header_length);
       std::string body = data.substr(header_length);

       // Стартовая строка: METHOD TARGET HTTP/1.1
       std::istringstream request_line(headers.substr(0, headers.find("\r\n")));
       std::string method, target;
       request_line >> method >> target;
       target = target.substr(0, target.find('?'));
       std::cout << method << ' ' << target << '\n';

       size_t content_length = 0;
       std::istringstream header_lines(headers);
       for (std::string line; std::getline(header_lines, line);) {
         std::string lower(line);
         std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
         if (lower.rfind("content-length:", 0) == 0) {
           content_length = std::stoul(line.substr(line.find(':') + 1));
         }
       }

       std::string response;
       if (content_length > kMaxBodySize) {
         response = makeResponse("413 Payload Too Large", jsonError("Image is larger than 10 MB"));
       } else {
         // Дочитываем тело, если оно не пришло вместе с заголовками
         if (body.size() < content_length) {
           std::string rest(content_length - body.size(), '\0');
           read(*socket, buffer(rest));
           body += rest;
         }
         body.resize(content_length);
         response = getLogic(method, target, body);
       }
       std::cout << response.substr(0, response.find("\r\n")) << '\n';
       write(*socket, buffer(response));
     } catch (std::exception& e) {
       std::cerr << "Exception in thread: " << e.what() << '\n';
       boost::system::error_code ignored;
       write(*socket, buffer(makeResponse("500 Internal Server Error", jsonError(e.what()))), ignored);
     }
   }

