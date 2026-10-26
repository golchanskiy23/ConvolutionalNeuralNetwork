import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
from PIL import Image
from tempfile import TemporaryDirectory


def parse_args():
    parser = argparse.ArgumentParser(description='Дообучение ResNet-18 и экспорт модели для C++ сервера')
    # Предположим, что данные располагаются в каталогах 'train' и 'valid'
    parser.add_argument('--data-dir', default='arch', help='каталог с подкаталогами train/ и valid/')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--out-dir', default=os.path.join('..', 'server', 'model'),
                        help='куда сохранить model.pt и classes.txt')
    return parser.parse_args()


# Преобразования изображений для нормализации
# (сервер повторяет 'valid' в ImageClassifier::preprocess, при изменении править в обоих местах)
data_transforms = {
    # Аугментации только для обучения
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Обучение модели
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


# Оценка на test. Учитываются только классы, на которых модель обучалась
# (в arch/test 525 классов, а в arch/train - 2)
def evaluate_test(model, test_dir, class_names):
    test_dataset = datasets.ImageFolder(test_dir, data_transforms['valid'])
    class_index = {name: i for i, name in enumerate(class_names)}
    indices = [i for i, (_, label) in enumerate(test_dataset.samples)
               if test_dataset.classes[label] in class_index]
    if not indices:
        print('No test images for trained classes')
        return

    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_dataset, indices), batch_size=16, num_workers=4)
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in loader:
            # Индексы классов test -> индексы классов train
            labels = torch.tensor([class_index[test_dataset.classes[l]] for l in labels.tolist()])
            outputs = model(inputs.to(device))
            corrects += torch.sum(outputs.argmax(1).cpu() == labels).item()

    print(f'Test Acc: {corrects / len(indices):.4f} ({corrects}/{len(indices)})')


# Сохранение модели в TorchScript и списка классов для C++ сервера
def export_model(model, class_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = model.cpu().eval()

    model_path = os.path.join(out_dir, 'model.pt')
    torch.jit.script(model).save(model_path)

    # Порядок строк = индекс выхода модели (ImageFolder сортирует классы по имени)
    classes_path = os.path.join(out_dir, 'classes.txt')
    with open(classes_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names) + '\n')

    print(f'Model saved to {model_path}, classes ({len(class_names)}) to {classes_path}')


if __name__ == '__main__':
    args = parse_args()

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    # Загрузка предварительно обученной модели ResNet-18
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    # Замораживаем свёрточную часть: обучается только новый fc, backward через всю сеть не нужен
    for param in model_ft.parameters():
        param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Обучение только последнего слоя (классификатора)
    optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

    # Уменьшение скорости обучения на 7 эпохе на 10x
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Обучение и оценка модели
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)

    test_dir = os.path.join(args.data_dir, 'test')
    if os.path.isdir(test_dir):
        evaluate_test(model_ft, test_dir, class_names)

    export_model(model_ft, class_names, args.out_dir)


