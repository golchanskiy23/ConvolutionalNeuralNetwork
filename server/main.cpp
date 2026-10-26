#include <iostream>
#include "WebServer.h"
#include "ImageClassification.h"

// Использование: ./server [model.pt] [classes.txt] [port]
int main(int argc, char* argv[])
{
  const std::string modelPath = argc > 1 ? argv[1] : "model/model.pt";
  const std::string classesPath = argc > 2 ? argv[2] : "model/classes.txt";
  const short port = argc > 3 ? static_cast<short>(std::stoi(argv[3])) : 8001;

  try
  {
    ImageClassifier classifier(modelPath, classesPath);
    std::cout << "Loaded model " << modelPath << " (" << classifier.numClasses() << " classes)\n";

    io_context service;
    HTTPServer server(service, port, classifier);
    std::cout << "Listening on port " << port << ", POST /predict" << std::endl;
    service.run();
  }
  catch(std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
