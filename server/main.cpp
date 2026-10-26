#include <csignal>
#include <iostream>
#include "WebServer.h"
#include "ImageClassification.h"

// Использование: ./server [model.pt] [classes.txt] [port] [threads]
int main(int argc, char* argv[])
{
  try
  {
    const std::string modelPath = argc > 1 ? argv[1] : "model/model.pt";
    const std::string classesPath = argc > 2 ? argv[2] : "model/classes.txt";
    const auto port = static_cast<unsigned short>(argc > 3 ? std::stoi(argv[3]) : 8001);
    const size_t threads = argc > 4 ? std::stoul(argv[4]) : 2;

    ImageClassifier classifier(modelPath, classesPath);
    std::cout << "Loaded model " << modelPath << " (" << classifier.numClasses() << " classes)\n";

    io_context service;
    HTTPServer server(service, port, classifier, threads);

    // Корректное завершение по Ctrl+C и docker stop
    signal_set signals(service, SIGINT, SIGTERM);
    signals.async_wait([&service](const boost::system::error_code&, int signal) {
      std::cout << "Signal " << signal << ", shutting down" << std::endl;
      service.stop();
    });

    std::cout << "Listening on port " << port << " with " << threads << " inference threads, POST /predict"
              << std::endl;
    service.run();
  }
  catch(std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
