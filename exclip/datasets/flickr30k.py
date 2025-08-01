import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..utils.flickr import get_annotations, get_sentence_data


class FlickrDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, split="all") -> None:
        self.split = split
        self.dataset_dir = dataset_dir
        self.entities_dir = os.path.join(dataset_dir, "flickr30k_entities")
        self.sentences_path = os.path.join(self.entities_dir, "Sentences")
        self.annotations_path = os.path.join(self.entities_dir, "Annotations")
        self.images_path = os.path.join(dataset_dir, "flickr30k_images", "images")
        assert len(os.listdir(self.annotations_path)) == len(
            os.listdir(self.sentences_path)
        )

        self.sentences = os.listdir(self.sentences_path)
        self.annotations = os.listdir(self.annotations_path)

        assert split in ["all", "train", "val", "test"]
        self.data_ids = self.get_data_ids(split=split)

        self.data = self.build_dataset()

    def __getitem__(self, id):
        instance = self.data[id]
        instance["image"] = self._load_image(instance["id"])
        instance["image_id"] = instance["id"]
        return instance

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, id: str):
        img = Image.open(os.path.join(self.images_path, id + ".jpg")).convert("RGB")
        return img

    def get_data_ids(self, split):
        ids_sentences = [id_txt.split(".")[0] for id_txt in self.sentences]
        ids_annotations = [id_xml.split(".")[0] for id_xml in self.annotations]
        if split == "all":
            return np.array(list(set(ids_annotations + ids_sentences)))
        else:
            split_path = os.path.join(self.entities_dir, split + ".txt")
            return np.loadtxt(split_path, dtype=str)

    def build_dataset(self):
        data = []
        exclude_counter = 0
        max_length = 77
        for idx in tqdm(self.data_ids):
            cpts = get_sentence_data(os.path.join(self.sentences_path, f"{idx}.txt"))
            boxes = get_annotations(os.path.join(self.annotations_path, f"{idx}.xml"))
            for cpt in cpts:
                if len(cpt["sentence"].split(" ")) > max_length:
                    exclude_counter += 1
                    continue
                instance = {"text": cpt["sentence"], "id": idx, "phrases": []}
                for phrase in cpt["phrases"]:
                    pid = phrase["phrase_id"]
                    if pid in boxes["boxes"]:
                        instance["phrases"].append(
                            {
                                "phrase": phrase["phrase"],
                                "phrase_id": phrase["phrase_id"],
                                "boxes": boxes["boxes"][pid],
                                "type": phrase["phrase_type"],
                            }
                        )
                data.append(instance)
        print(
            f"Excluded {exclude_counter} captions due to max length restriction of {max_length}"
        )
        return data


def flickr_collate_fn(data):
    captions = []
    images = []
    phrases = []
    ids = []
    for _tuple in data:
        captions.append(_tuple["text"])
        phrases.append(_tuple["phrases"])
        ids.append(_tuple["id"])
        images.append(_tuple["image"])
    data = {"text": captions, "images": images, "phrases": phrases, "ids": ids}
    return data
