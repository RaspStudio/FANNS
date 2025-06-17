import json
import os
import random

import numpy as np
from PIL import Image


def dirichlet_split(num_clients, beta=None):
    if beta == None:
        beta = 0.5
    return np.random.dirichlet([1] * num_clients, 1)


def clean_corrupt_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()
        except (IOError, SyntaxError):
            os.remove(file_path)


def save_distribution_dir(data: dict, output_path, dir_name):
    dir_path = os.path.join(output_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    count = 0
    for key in data.keys():
        if key == 'path':
            continue
        output = {}
        output['path'] = data['path']
        output['data'] = data[f'client_{count}']['data']
        output_p = os.path.join(dir_path, f'client_{count}.json')

        with open(output_p, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=4)
        count += 1
        print(f"save to {output_p}")


def gather_imagenet_images(imagenet_root_dir):

    image_extensions = (".jpg", ".png", ".jpeg")
    subfolders = [f for f in os.listdir(imagenet_root_dir) if os.path.isdir(
        os.path.join(imagenet_root_dir, f))]
    images = []
    for class_idx, folder in enumerate(subfolders):
        path = os.path.join(imagenet_root_dir, folder)
        clean_corrupt_images(path)

        sub_images = [img for img in os.listdir(
            path) if img.lower().endswith(image_extensions)]
        random.shuffle(sub_images)
        full_paths = [os.path.join(folder, img) for img in sub_images]
        images.append(full_paths)
    return images


def gather_iNatrualist_images(inaturalist_root_dir):

    subfolders = [f for f in os.listdir(inaturalist_root_dir) if os.path.isdir(
        os.path.join(inaturalist_root_dir, f))]
    for floder in subfolders:
        min_floders = [f for f in os.listdir(os.path.join(inaturalist_root_dir, floder)) if os.path.isdir(
            os.path.join(os.path.join(inaturalist_root_dir, floder), f))]
        for min in min_floders:
            clean_corrupt_images(os.path.join(
                os.path.join(inaturalist_root_dir, floder), min))
    categories = {}

    path = ["val2018", "train2018"]
    for file in path:
        file_path = "../data/i-Naturalist/" + file + ".json"
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        for idx in range(len(data['annotations'])):
            message = data['annotations'][idx]
            categorie_id = message['category_id']
            if categorie_id in categories:
                categories[categorie_id].append(
                    data['images'][idx]['file_name'])
            else:
                categories[categorie_id] = [data['images'][idx]['file_name']]

    images_collection = list(categories.values())
    print(len(images_collection))
    return images_collection


def gather_Marcov1_texts(Macrov1_root_dir):

    text_extensions = (".txt")
    texts = [text for text in os.listdir(
        Macrov1_root_dir) if text.lower().endswith(text_extensions)]
    random.shuffle(texts)
    return [texts]


def gather_sent140_texts(sent140_root_dir):
    with open(sent140_root_dir, "r") as f:
        user_data = json.load(f)
    texts = []
    for user in user_data.keys():
        texts.append(user_data[user])
    return text


def distribute_label(root_dir, num_clients, distribution_matrix, images_collection, num):
    data = {'path': root_dir}
    for i in range(num_clients):
        data[f'client_{i}'] = {'data': []}

    num_images = num
    split_counts = (distribution_matrix[0] * num_images).astype(int)
    missing = num_images - np.sum(split_counts)
    split_counts[-1] += missing
    print(split_counts)

    index = 0
    for class_idx, images in enumerate(images_collection):
        if index >= num_clients:
            break
        data[f'client_{index}']['data'].extend(
            images[:split_counts[index] - len(data[f'client_{index}']['data'])])
        if len(data[f'client_{index}']['data']) >= split_counts[index]:
            index += 1
            print(f"index is {index}")

    for i in range(num_clients):
        data[f'client_{i}']['num'] = len(data[f'client_{i}']['data'])

    return data


def distribute_Marco(root_dir, num_clients, distribution_matrix, images_collection, num):
    data = {'path': root_dir}
    for i in range(num_clients):
        data[f'client_{i}'] = {'data': []}

    num_images = num
    split_counts = (distribution_matrix[0] * num_images).astype(int)
    missing = num_images - np.sum(split_counts)
    split_counts[-1] += missing

    index = 0

    for i in range(num_clients):
        data[f'client_{i}']['data'].extend(
            images_collection[0][index:index + split_counts[i]])
        index += split_counts[i]

    for i in range(num_clients):
        data[f'client_{i}']['num'] = len(data[f'client_{i}']['data'])

    return data


def process_dataset(dataset_name, gather_function, data_num, data_dir, num_clients, output_path, dir_name, beta=None, distribute_func=distribute_label):

    print(f"Processing {dataset_name} dataset...")
    images_collection = gather_function(data_dir)

    distribution_matrix = dirichlet_split(num_clients, beta)

    client_data = distribute_func(
        data_dir, num_clients, distribution_matrix, images_collection, data_num)
    save_distribution_dir(client_data, output_path, dir_name)

    print(f"{dataset_name} dataset processing completed!")


if __name__ == "__main__":
    # Dataset Distribution
    process_dataset(dataset_name="imagenet",
                    gather_function=gather_imagenet_images,
                    data_num=1000,
                    data_dir="../data/imagenet/train",
                    num_clients=4,
                    output_path="../distribution/",
                    dir_name="test",
                    beta=0.5,
                    distribute_func=distribute_label)
