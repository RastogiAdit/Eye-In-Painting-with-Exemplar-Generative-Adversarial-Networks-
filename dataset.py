import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from PIL import ImageFile
import json
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES
from PIL import Image, ImageDraw
import copy


class ExGANDataset(Dataset):
    def __init__(self, root_dir, transform=None, exemplar_dir=None, image_size=128):
        """
        Args:
            root_dir (string): Directory with all the training images
            transform (callable, optional): Optional transform to be applied
            exemplar_pairs (bool): If True, expects exemplar_dir with paired images
            exemplar_dir (string): Directory with exemplar images (if using pairs)
            image_size (int): Target size for image transformations
        """
        self.root_dir = root_dir
        self.exemplar_dir = exemplar_dir
        self.image_size = image_size

        # Get list of image files
        self.image_files = [
            f for f in os.listdir(root_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]

        # Filter images that exist in both directories
        valid_images = []
        for img_file in self.image_files:
            exemplar_path = os.path.join(exemplar_dir, img_file)
            if os.path.exists(exemplar_path):
                valid_images.append(img_file)

        self.image_files = valid_images
        # self.image_files = valid_images

        # Set up transformations
        self.transform = transform if transform else self.get_default_transforms()

        celeb_json = "../data/celeb_id_aligned/data.json"

        with open(celeb_json, "r") as f:
            data = json.load(f)

        self.eyes = {}
        for key, values in data.items():
            for i in values:
                if (
                    "box_left" in i
                    and "box_right" in i
                    and "eye_left" in i
                    and "eye_right" in i
                ):
                    self.eyes[i["filename"]] = i

        # Filter images that have valid eye parameters
        valid_images = []
        for img_file in self.image_files:
            if img_file in self.eyes:
                valid_images.append(img_file)

        self.image_files = valid_images

        print(len(self.image_files))

    def get_default_transforms(self):
        """Returns default transformations for ExGAN training"""
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def create_image_paches_for_eyes(self, image, eyes):
        # # Convert tensor to PIL Image first
        # if isinstance(image, torch.Tensor):
        #     image = transforms.ToPILImage()(image)

        # # Convert PIL image to numpy array
        # img_array = np.array(image)

        left_x1 = int(eyes["eye_left"]["x"])
        left_y1 = int(eyes["eye_left"]["y"])
        right_x1 = int(eyes["eye_right"]["x"])
        right_y1 = int(eyes["eye_right"]["y"])

        left_w = int(eyes["box_left"]["w"])
        left_h = int(eyes["box_left"]["h"])
        right_w = int(eyes["box_right"]["w"])
        right_h = int(eyes["box_right"]["h"])

        draw = ImageDraw.Draw(image)

        # Draw box around left eye using center coordinates
        left_box_x1 = left_x1 - left_w // 2  # Left edge
        left_box_y1 = left_y1 - left_h // 2  # Top edge
        left_box_x2 = left_x1 + left_w // 2  # Right edge
        left_box_y2 = left_y1 + left_h // 2  # Bottom edge

        # Draw box around right eye using center coordinates
        right_box_x1 = right_x1 - right_w // 2  # Left edge
        right_box_y1 = right_y1 - right_h // 2  # Top edge
        right_box_x2 = right_x1 + right_w // 2  # Right edge
        right_box_y2 = right_y1 + right_h // 2  # Bottom edge

        # Draw rectangles
        draw.rectangle(
            [left_box_x1, left_box_y1, left_box_x2, left_box_y2], fill="gray"
        )
        draw.rectangle(
            [right_box_x1, right_box_y1, right_box_x2, right_box_y2], fill="gray"
        )

        # image.save("test.png")

        return image

        import pdb

        pdb.set_trace()
        # Fill eye regions with gray (128)
        img_array[left_y1 : left_y1 + left_h, left_x1 : left_x1 + left_w] = 128
        img_array[right_y1 : right_y1 + right_h, right_x1 : right_x1 + right_w] = 128

        # Convert back to PIL Image
        pil_image = Image.fromarray(img_array.astype("uint8"))
        pil_image.save("test.png")
        # Convert back to tensor
        return transforms.ToTensor()(pil_image)

    def __getitem__(self, idx):
        # Load main image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            masked_image = self.create_image_paches_for_eyes(
                copy.deepcopy(image), self.eyes[self.image_files[idx]]
            )

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.image_files) - 1))

        # Load corresponding exemplar image
        exemplar_path = os.path.join(self.exemplar_dir, self.image_files[idx])
        exemplar = Image.open(exemplar_path).convert("RGB")

        # Apply same transformation to both images
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        exemplar = self.transform(exemplar)

        random.seed(seed)
        torch.manual_seed(seed)
        masked_image = self.transform(masked_image)

        return image, masked_image, exemplar


def get_data_loaders(
    data_dir,
    batch_size=16,
    image_size=128,
    num_workers=4,
    exemplar_dir=None,
    validation_split=0.1,
):
    """
    Creates train and validation dataloaders for ExGAN training

    Args:
        data_dir (string): Directory with training images
        batch_size (int): Batch size for training
        image_size (int): Target image size
        num_workers (int): Number of workers for data loading
        exemplar_dir (string): Optional directory with exemplar images
        validation_split (float): Fraction of data to use for validation

    Returns:
        train_loader, val_loader: DataLoader instances
    """
    # Check if we have paired exemplars
    exemplar_pairs = exemplar_dir is not None

    # Create full dataset
    full_dataset = ExGANDataset(
        root_dir=data_dir,
        exemplar_pairs=exemplar_pairs,
        exemplar_dir=exemplar_dir,
        image_size=image_size,
    )

    # Split into train and validation
    val_size = int(validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    dataset = ExGANDataset(
        root_dir="../data/celeb_id_aligned", exemplar_dir="../data/celeb_id_raw"
    )

    print(len(dataset))
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    for i, (raw_img, aligned_img) in enumerate(dataloader):
        print(raw_img.shape, aligned_img.shape)
        pass
