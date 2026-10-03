from torchvision import transforms


augmentation_list = {

    "No Augmentation": transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ]),

    "Basic": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ]),

    "Color": transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor()
    ]),

    "Crop": transforms.Compose([
        transforms.RandomResizedCrop(
            150,
            scale=(0.7, 1.0)
        ),
        transforms.ToTensor()
    ]),

    "Full without Color": transforms.Compose([
        transforms.RandomResizedCrop(
            150,
            scale=(0.7, 1.0)
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15)
        )
    ]),

    "Full + Basic": transforms.Compose([
        transforms.RandomResizedCrop(
            150,
            scale=(0.7, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15)
        )
    ]),

    "Full + Basic + Color": transforms.Compose([
        transforms.RandomResizedCrop(
            150,
            scale=(0.7, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15)
        )
    ]),

    "Full": transforms.Compose([
        transforms.RandomResizedCrop(
            150,
            scale=(0.7, 1.0)
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15)
        )
    ])
}


augmentation_count = {
    "No Augmentation": 0,
    "Basic": 2,
    "Color": 1,
    "Crop": 1,
    "Full without Color": 3,
    "Full + Basic": 5,
    "Full + Basic + Color": 6,
    "Full": 4
}


eval_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])