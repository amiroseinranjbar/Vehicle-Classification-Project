import os
import hashlib
import shutil
from collections import Counter

from PIL import Image
import matplotlib.pyplot as plt


DATASET1 = "dataset_n"
DATASET2 = "datasetv2_TrainUclean_n"

CLASSES = [
    "ambulance",
    "autobus",
    "kamyun",
    "kamyunet",
    "minibus",
    "savari",
    "taxi",
    "vanet"
]

DATA_PATHS = {
    "dataset1_train": os.path.join(DATASET1, "train"),
    "dataset1_test": os.path.join(DATASET1, "test"),
    "dataset1_unclean": os.path.join(DATASET1, "unclean"),
    "dataset2_train": os.path.join(DATASET2, "train"),
    "dataset2_unclean": os.path.join(DATASET2, "unclean")
}

MERGED_PATH = os.path.join("dataset", "merged_dataset")
MERGED_TRAIN_PATH = os.path.join(MERGED_PATH, "train")


def get_md5(file_path):

    md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(4096),
            b""
        ):
            md5.update(chunk)

    return md5.hexdigest()


def get_files(folder_path):

    files = []

    if not os.path.exists(folder_path):
        return files

    for folder in os.listdir(folder_path):

        folder_path2 = os.path.join(
            folder_path,
            folder
        )

        if not os.path.isdir(folder_path2):
            continue

        for image_name in os.listdir(folder_path2):

            image_path = os.path.join(
                folder_path2,
                image_name
            )

            if not os.path.isfile(image_path):
                continue

            try:

                file_hash = get_md5(image_path)

                files.append(
                    (
                        file_hash,
                        image_path,
                        folder
                    )
                )

            except:

                pass

    return files


def check_corrupted_images():

    print("\nCORRUPTED IMAGE CHECK")
    print("=" * 50)

    for name, path in DATA_PATHS.items():

        corrupted = []

        for folder in os.listdir(path):

            folder_path = os.path.join(
                path,
                folder
            )

            if not os.path.isdir(folder_path):
                continue

            for image_name in os.listdir(folder_path):

                image_path = os.path.join(
                    folder_path,
                    image_name
                )

                try:

                    image = Image.open(image_path)
                    image.verify()

                except:

                    corrupted.append(image_path)

        print(f"{name}: {len(corrupted)} corrupted images")

        for image_path in corrupted:
            print(image_path)


def find_unclean_duplicates():

    test_files = get_files(
        DATA_PATHS["dataset1_test"]
    )

    train1_files = get_files(
        DATA_PATHS["dataset1_train"]
    )

    train2_files = get_files(
        DATA_PATHS["dataset2_train"]
    )

    test_hashes = {
        file_hash: label
        for file_hash, path, label in test_files
    }

    train1_hashes = {
        file_hash: label
        for file_hash, path, label in train1_files
    }

    train2_hashes = {
        file_hash: label
        for file_hash, path, label in train2_files
    }

    files_to_delete = []
    label_conflicts = []

    unclean_paths = [
        DATA_PATHS["dataset1_unclean"],
        DATA_PATHS["dataset2_unclean"]
    ]

    for unclean_path in unclean_paths:

        for file_hash, path, label in get_files(
            unclean_path
        ):

            if file_hash in test_hashes:

                files_to_delete.append(
                    (path, "duplicate with TEST")
                )

                if label != test_hashes[file_hash]:

                    label_conflicts.append(
                        (
                            path,
                            label,
                            "TEST",
                            test_hashes[file_hash]
                        )
                    )

            elif file_hash in train1_hashes:

                files_to_delete.append(
                    (
                        path,
                        "duplicate with dataset1 TRAIN"
                    )
                )

                if label != train1_hashes[file_hash]:

                    label_conflicts.append(
                        (
                            path,
                            label,
                            "dataset1 TRAIN",
                            train1_hashes[file_hash]
                        )
                    )

            elif file_hash in train2_hashes:

                files_to_delete.append(
                    (
                        path,
                        "duplicate with dataset2 TRAIN"
                    )
                )

                if label != train2_hashes[file_hash]:

                    label_conflicts.append(
                        (
                            path,
                            label,
                            "dataset2 TRAIN",
                            train2_hashes[file_hash]
                        )
                    )

    print("\nUNCLEAN DUPLICATE CHECK")
    print("=" * 50)

    print(
        "Files to delete:",
        len(files_to_delete)
    )

    print(
        "Label conflicts:",
        len(label_conflicts)
    )

    for path, reason in files_to_delete:

        print(
            "DELETE:",
            path,
            "|",
            reason
        )

    for conflict in label_conflicts:

        print(
            "CONFLICT:",
            conflict
        )

    for path, reason in files_to_delete:

        if os.path.exists(path):

            os.remove(path)

            print(
                "Deleted:",
                path
            )


def check_cross_dataset_duplicates():

    all_files = {}

    for name, path in DATA_PATHS.items():

        all_files[name] = get_files(path)

    hashes = {

        name: {
            file_hash: path
            for file_hash, path, label in files
        }

        for name, files in all_files.items()
    }

    names = list(hashes.keys())

    print("\nCROSS DATASET DUPLICATE CHECK")
    print("=" * 50)

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            name1 = names[i]
            name2 = names[j]

            duplicates = (
                set(hashes[name1].keys())
                &
                set(hashes[name2].keys())
            )

            print(
                name1,
                "<->",
                name2,
                ":",
                len(duplicates)
            )


def check_image_sizes():

    print("\nIMAGE SIZE CHECK")
    print("=" * 50)

    for name, path in DATA_PATHS.items():

        sizes = Counter()

        for folder in os.listdir(path):

            folder_path = os.path.join(
                path,
                folder
            )

            if not os.path.isdir(folder_path):
                continue

            for image_name in os.listdir(folder_path):

                image_path = os.path.join(
                    folder_path,
                    image_name
                )

                try:

                    image = Image.open(image_path)

                    sizes[image.size] += 1

                except:

                    pass

        print(f"\n{name}")

        for size, count in sizes.most_common():

            print(
                f"{size}: {count}"
            )


def show_small_images(threshold=120):

    print("\nSMALL IMAGE CHECK")
    print("=" * 50)

    for name, path in DATA_PATHS.items():

        for folder in os.listdir(path):

            folder_path = os.path.join(
                path,
                folder
            )

            if not os.path.isdir(folder_path):
                continue

            shown = 0

            for image_name in os.listdir(folder_path):

                image_path = os.path.join(
                    folder_path,
                    image_name
                )

                try:

                    image = Image.open(image_path)

                    width, height = image.size

                    if (
                        width < threshold
                        or
                        height < threshold
                    ):

                        print(
                            f"{name} | "
                            f"{folder} | "
                            f"{width}x{height}"
                        )

                        plt.figure(figsize=(4, 4))

                        plt.imshow(image)

                        plt.title(
                            f"{name} | {folder}\n"
                            f"{width} x {height}"
                        )

                        plt.axis("off")

                        plt.show()

                        shown += 1

                    if shown == 2:
                        break

                except:

                    pass


def count_images():

    all_counts = {}

    print("\nCLASS DISTRIBUTION")
    print("=" * 50)

    for name, path in DATA_PATHS.items():

        counts = {}

        for folder in sorted(
            os.listdir(path)
        ):

            folder_path = os.path.join(
                path,
                folder
            )

            if not os.path.isdir(folder_path):
                continue

            count = 0

            for image_name in os.listdir(
                folder_path
            ):

                image_path = os.path.join(
                    folder_path,
                    image_name
                )

                if os.path.isfile(image_path):

                    count += 1

            counts[folder] = count

        all_counts[name] = counts

        print(f"\n{name}")

        for folder, count in counts.items():

            print(
                f"{folder}: {count}"
            )

        plt.figure(figsize=(10, 6))

        plt.bar(
            counts.keys(),
            counts.values()
        )

        plt.xlabel("Class")
        plt.ylabel("Number of Images")

        plt.title(
            f"{name} - Number of Images per Class"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        plt.savefig(
            f"{name}_class_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    return all_counts


def merge_clean_train_data():

    os.makedirs(
        MERGED_TRAIN_PATH,
        exist_ok=True
    )

    train_paths = [
        DATA_PATHS["dataset1_train"],
        DATA_PATHS["dataset2_train"]
    ]

    for train_path in train_paths:

        for folder in os.listdir(train_path):

            source_folder = os.path.join(
                train_path,
                folder
            )

            if not os.path.isdir(source_folder):
                continue

            destination_folder = os.path.join(
                MERGED_TRAIN_PATH,
                folder
            )

            os.makedirs(
                destination_folder,
                exist_ok=True
            )

            for image_name in os.listdir(
                source_folder
            ):

                source = os.path.join(
                    source_folder,
                    image_name
                )

                destination = os.path.join(
                    destination_folder,
                    image_name
                )

                if os.path.isfile(source):

                    shutil.copy2(
                        source,
                        destination
                    )


def merge_unclean_data():

    unclean_paths = [
        DATA_PATHS["dataset1_unclean"],
        DATA_PATHS["dataset2_unclean"]
    ]

    for unclean_path in unclean_paths:

        for folder in CLASSES:

            source_folder = os.path.join(
                unclean_path,
                folder
            )

            if not os.path.exists(source_folder):
                continue

            destination_folder = os.path.join(
                MERGED_TRAIN_PATH,
                folder
            )

            os.makedirs(
                destination_folder,
                exist_ok=True
            )

            for image_name in os.listdir(
                source_folder
            ):

                source = os.path.join(
                    source_folder,
                    image_name
                )

                destination = os.path.join(
                    destination_folder,
                    "unclean_" + image_name
                )

                if os.path.isfile(source):

                    shutil.copy2(
                        source,
                        destination
                    )


def create_merged_dataset():

    if os.path.exists(MERGED_TRAIN_PATH):

        shutil.rmtree(
            MERGED_TRAIN_PATH
        )

    os.makedirs(
        MERGED_TRAIN_PATH,
        exist_ok=True
    )

    merge_clean_train_data()
    merge_unclean_data()

    print("\nMerged dataset created:")
    print(MERGED_TRAIN_PATH)


if __name__ == "__main__":

    check_corrupted_images()

    find_unclean_duplicates()

    check_cross_dataset_duplicates()

    check_image_sizes()

    count_images()

    create_merged_dataset()

