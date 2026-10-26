#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include "WebServer.h"
#include "ImageClassification.h"

using namespace boost::asio;

   // Максимальный размер заголовков и тела запроса (изображения)
   static const size_t kMaxHeaderSize = 64 * 1024;
   static const size_t kMaxBodySize = 10 * 1024 * 1024;
   // Сколько ждём, пока клиент пришлёт запрос целиком
   static const std::chrono::seconds kReadTimeout(15);

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

   ThreadPool::ThreadPool(size_t num_threads)
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
           }
           try {
             task();
           } catch (std::exception& e) {
             std::cerr << "Exception in thread: " << e.what() << '\n';
           }
           {
             std::lock_guard<std::mutex> lock(mutex_);
             --busy_;
           }
         }
       });
     }
   }

   bool ThreadPool::try_enqueue(std::function<void()> task) {
     {
       std::lock_guard<std::mutex> lock(mutex_);
       if (stop_ || busy_ >= threads_.size()) {
         return false;
       }
       ++busy_;
       tasks_.push(std::move(task));
     }
     condition_.notify_one();
     return true;
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

   HTTPServer::HTTPServer(io_context& io_context, unsigned short port, ImageClassifier& classifier, size_t num_threads)
     : acceptor_(io_context, ip::tcp::endpoint(ip::tcp::v4(), port)), io_context_(io_context),
       classifier_(classifier), pool(num_threads)
   {
     start_accept();
   }

   // business logic function
   std::string HTTPServer::getLogic(const std::string& method, const std::string& target, const std::string& body)
   {
     if (target == "/health") {
       return makeResponse("200 OK", "{\"status\": \"ok\", \"classes\": " +
                                     std::to_string(classifier_.numClasses()) + "}\n");
     }
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
     } catch (std::exception& e) {
       std::cerr << "Inference error: " << e.what() << '\n';
       return makeResponse("500 Internal Server Error", jsonError(e.what()));
     }
   }

   void HTTPServer::start_accept() {
     acceptor_.async_accept([this](const boost::system::error_code& error, ip::tcp::socket socket) {
       if (!error) {
         std::make_shared<Session>(std::move(socket), *this)->start();
       }
       if (acceptor_.is_open()) {
         start_accept();
       }
     });
   }

   Session::Session(ip::tcp::socket socket, HTTPServer& server)
     : socket_(std::move(socket)), server_(server), deadline_(server.io_context_), buf_(kMaxHeaderSize)
   {
   }

   void Session::start() {
     // Медленный или молчащий клиент не должен держать соединение вечно
     deadline_.expires_after(kReadTimeout);
     deadline_.async_wait([self = shared_from_this()](const boost::system::error_code& error) {
       if (!error) {
         boost::system::error_code ignored;
         self->socket_.close(ignored);
       }
     });
     read_headers();
   }

   void Session::read_headers() {
     async_read_until(socket_, buf_, "\r\n\r\n",
       [self = shared_from_this()](const boost::system::error_code& ec, size_t header_length) {
         if (ec == error::not_found) {
           self->write_response(makeResponse("431 Request Header Fields Too Large", jsonError("Headers too large")));
           return;
         }
         if (ec) return;  // клиент закрыл соединение или истёк таймаут

         std::string headers(buffers_begin(self->buf_.data()), buffers_begin(self->buf_.data()) + header_length);
         self->buf_.consume(header_length);

         // Стартовая строка: METHOD TARGET HTTP/1.1
         std::istringstream request_line(headers.substr(0, headers.find("\r\n")));
         request_line >> self->method_ >> self->target_;
         self->target_ = self->target_.substr(0, self->target_.find('?'));

         size_t content_length = 0;
         std::istringstream header_lines(headers);
         try {
           for (std::string line; std::getline(header_lines, line);) {
             std::string lower(line);
             std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
             if (lower.rfind("content-length:", 0) == 0) {
               content_length = std::stoul(line.substr(line.find(':') + 1));
             }
           }
         } catch (std::exception&) {
           self->write_response(makeResponse("400 Bad Request", jsonError("Invalid Content-Length")));
           return;
         }

         if (content_length > kMaxBodySize) {
           self->write_response(makeResponse("413 Payload Too Large", jsonError("Image is larger than 10 MB")));
           return;
         }
         self->read_body(content_length);
       });
   }

   void Session::read_body(size_t content_length) {
     // Часть тела могла прийти вместе с заголовками
     size_t buffered = std::min(buf_.size(), content_length);
     body_.assign(buffers_begin(buf_.data()), buffers_begin(buf_.data()) + buffered);
     buf_.consume(buffered);
     if (body_.size() == content_length) {
       process();
       return;
     }

     size_t offset = body_.size();
     body_.resize(content_length);
     async_read(socket_, buffer(&body_[offset], content_length - offset),
       [self = shared_from_this()](const boost::system::error_code& error, size_t) {
         if (!error) self->process();
       });
   }

   void Session::process() {
     deadline_.cancel();

     // Инференс - в пуле потоков, всё остальное отвечаем сразу из io_context
     bool needs_inference = method_ == "POST" && target_ == "/predict" && !body_.empty();
     if (!needs_inference) {
       write_response(server_.getLogic(method_, target_, body_));
       return;
     }

     bool accepted = server_.pool.try_enqueue([self = shared_from_this()] {
       std::string response = self->server_.getLogic(self->method_, self->target_, self->body_);
       // Сокетом пользуемся только из потока io_context
       post(self->server_.io_context_, [self, response = std::move(response)]() mutable {
         self->write_response(std::move(response));
       });
     });
     if (!accepted) {
       std::cerr << "Error: All threads are busy. Returning 503 Service Unavailable.\n";
       write_response(makeResponse("503 Service Unavailable", jsonError("All threads are busy")));
     }
   }

   void Session::write_response(std::string response) {
     deadline_.cancel();
     std::cout << method_ << ' ' << target_ << " -> " << response.substr(9, response.find("\r\n") - 9) << std::endl;
     response_ = std::move(response);
     async_write(socket_, buffer(response_),
       [self = shared_from_this()](const boost::system::error_code&, size_t) {
         boost::system::error_code ignored;
         self->socket_.shutdown(ip::tcp::socket::shutdown_both, ignored);
       });
   }
