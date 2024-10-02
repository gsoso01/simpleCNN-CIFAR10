import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np

def filter_and_remap_classes(dataset, classes, new_class_mapping):
    # Filter the dataset to include only the specified classes
    idx = [i for i, label in enumerate(dataset.targets) if label in classes]
    dataset.targets = np.array(dataset.targets)[idx].tolist()
    dataset.data = dataset.data[idx]
    # Remap to 0, 1, 2, etc
    dataset.targets = [new_class_mapping[label] for label in dataset.targets]
    
    return dataset

def load_datasets(batch_size=32):
    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)

    original_classes = [3, 4, 5]
    new_class_mapping = {3: 0, 4: 1, 5: 2}

    train_dataset = filter_and_remap_classes(train_dataset, original_classes, new_class_mapping)
    test_dataset = filter_and_remap_classes(test_dataset, original_classes, new_class_mapping)

    train_data, val_data = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), int(0.2*len(train_dataset))])

    # multiprocessing_context for MacOS, if in Linux, remove and be happy!
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, multiprocessing_context='fork')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context='fork')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context='fork')
    
    return train_loader, val_loader, test_loader
