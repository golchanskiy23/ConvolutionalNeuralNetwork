#ifndef IMAGE_CLASSIFICATION_IMAGECLASSIFICAION_H
#define IMAGE_CLASSIFICATION_IMAGECLASSIFICAION_H

#include <string>
#include <vector>
#include <torch/script.h>

struct Prediction {
  std::string className;
  float confidence;
};

class ImageClassifier {
  public:
   // modelPath - TorchScript-модель (model.pt), classesPath - имена классов по одному на строку (classes.txt)
   ImageClassifier(const std::string& modelPath, const std::string& classesPath);

   // imageBytes - содержимое файла изображения (JPEG/PNG/BMP/...)
   Prediction predict(const std::string& imageBytes);

   size_t numClasses() const { return class_names.size(); }

  private:
   torch::Tensor preprocess(const std::string& imageBytes);

   torch::jit::Module model;
   std::vector<std::string> class_names;
};

#endif //IMAGE_CLASSIFICATION_IMAGECLASSIFICAION_H
