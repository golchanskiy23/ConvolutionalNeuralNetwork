#include "ImageClassification.h"

#include <fstream>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace F = torch::nn::functional;

// Должно совпадать с data_transforms в training/classification.py
static const int64_t kImageSize = 224;
static const std::vector<double> kMean = {0.485, 0.456, 0.406};
static const std::vector<double> kStd = {0.229, 0.224, 0.225};

   ImageClassifier::ImageClassifier(const std::string& modelPath, const std::string& classesPath) {
     std::ifstream classes(classesPath);
     if (!classes) throw std::runtime_error("Cannot open classes file: " + classesPath);
     for (std::string line; std::getline(classes, line);) {
       if (!line.empty()) class_names.push_back(line);
     }
     if (class_names.empty()) throw std::runtime_error("No classes in " + classesPath);

     model = torch::jit::load(modelPath, torch::kCPU);
     model.eval();
   }

   torch::Tensor ImageClassifier::preprocess(const std::string& imageBytes) {
     int width = 0, height = 0, channels = 0;
     // Всегда 3 канала (RGB), как Image.convert('RGB') в ImageFolder
     std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
         stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(imageBytes.data()),
                               static_cast<int>(imageBytes.size()), &width, &height, &channels, 3),
         &stbi_image_free);
     if (!pixels) throw std::invalid_argument(std::string("Cannot decode image: ") + stbi_failure_reason());

     // HWC uint8 -> NCHW float [0, 1]
     torch::Tensor tensor = torch::from_blob(pixels.get(), {height, width, 3}, torch::kUInt8)
                                .permute({2, 0, 1})
                                .to(torch::kFloat32)
                                .div(255)
                                .unsqueeze(0);

     // transforms.Resize((224, 224)) - билинейная интерполяция со сглаживанием
     tensor = F::interpolate(tensor, F::InterpolateFuncOptions()
                                         .size(std::vector<int64_t>{kImageSize, kImageSize})
                                         .mode(torch::kBilinear)
                                         .align_corners(false)
                                         .antialias(true));

     // transforms.Normalize(mean, std)
     auto mean = torch::tensor(kMean, torch::kFloat32).view({1, 3, 1, 1});
     auto std = torch::tensor(kStd, torch::kFloat32).view({1, 3, 1, 1});
     return (tensor - mean) / std;
   }

   Prediction ImageClassifier::predict(const std::string& imageBytes) {
     torch::InferenceMode guard;
     torch::Tensor input = preprocess(imageBytes);

     torch::Tensor output = model.forward({input}).toTensor();
     torch::Tensor probabilities = torch::softmax(output, /*dim=*/1);
     auto [confidence, index] = probabilities.max(/*dim=*/1);

     int64_t predictedClassIndex = index.item<int64_t>();
     if (predictedClassIndex >= static_cast<int64_t>(class_names.size())) {
       throw std::runtime_error("Model output has more classes than classes file");
     }
     return {class_names[predictedClassIndex], confidence.item<float>()};
   }
