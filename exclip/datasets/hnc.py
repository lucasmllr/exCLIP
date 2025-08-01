import logging
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from ..utils.json_fns import load_json, save_json


class HNCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: dict,
        split: str,
        ans2label: str = "../meta_data/hnc_ans2label.json",
        label2ans: str = "../meta_data/hnc_label2ans.json",
        debugging: bool = False,
        return_bbox: bool = False,
        train_data_ratio: float = 1.0,
        create_ans_label_dicts: bool = False,
        only_correct_captions: bool = False,
    ) -> None:
        # list all assert statements here
        self.name = "HNCDataset"
        self.return_bbox = return_bbox
        logger = logging.getLogger("exclip")
        assert split in [
            "train",
            "valid",
            "test",
        ], f"Invalid split provided for {self.name}: {split}"

        # store dataset properties
        logger.info(f"preparing HNC {split} dataset")
        if only_correct_captions:
            logger.info("WARNING: only loading correct data samples, i.e. label==1")
        self.split = split
        hnc_config = cfg.datasets.hnc
        self.image_path = hnc_config.images
        if os.path.isfile(hnc_config.captions.get(split)):
            file_path = hnc_config.captions.get(split)
        else:
            file_path = hnc_config.captions.base + hnc_config.captions.get(split)

        if debugging and split == "train":
            file_path = "..path_to_debug/hnc/hnc_clean_strict_train_debug.pt"
            self.captions_raw = torch.load(file_path)
            logger.info(f"loaded {self.split} DEBUGGING captions file")
        elif "json" in hnc_config.captions.get(split):
            self.captions_raw = load_json(file_path)
        elif "pt" in hnc_config.captions.get(split):
            self.captions_raw = torch.load(file_path)
        logger.info(f"loaded {self.split} captions file")
        self.captions = self._create_caption_list(
            self.captions_raw, only_correct_captions=cfg.training.only_correct_captions
        )
        if self.split == "train":
            rnd_k_factor = train_data_ratio
            rnd_k = int(rnd_k_factor * len(self.captions))
            self.captions = random.choices(self.captions, k=rnd_k)
            logger.info(f"reduced train set to {rnd_k} with a factor of {rnd_k_factor}")
        if create_ans_label_dicts and (split == "train"):
            logger.info(f"creating hnc textual label dicts")
            self.ans2label_hnc, self.label2ans_hnc = self._create_hnc_ans_label_dicts()
            save_json("./meta_data/hnc_ans2label.json", self.ans2label_hnc)
            save_json("./meta_data/hnc_label2ans.json", self.label2ans_hnc)
        else:
            logger.info(f"loading hnc textual label dicts")
            self.ans2label_hnc = load_json(ans2label)
            self.label2ans_hnc = load_json(label2ans)

    def __getitem__(self, id):
        assert id < len(self), f"index {id} out of range of {len(self)} data samples"
        data_sample = self.captions[id]
        cpt = data_sample.get("caption")
        cpt = "" if not cpt else cpt
        img_id = data_sample.get("img_id")
        img_path = f"{self.image_path}/{img_id}"
        img_path = img_path if "jpg" in img_path else img_path + ".jpg"
        image = Image.open(img_path)

        itm_label = int(data_sample.get("label"))
        textual_label = data_sample.get("textual_label")
        if not textual_label:
            textual_id = -100
        else:
            textual_id = self.ans2label_hnc.get(textual_label, -100)

        cpt_type = data_sample.get("type")
        cpt_id = data_sample.get("cpt_id")

        if self.return_bbox:
            return {
                "text": cpt,
                "image": image,
                "itm_label": itm_label,
                "textual_id": textual_id,
                "cpt_type": cpt_type,
                "cpt_id": cpt_id,
                "image_id": img_id,
                "bboxes": data_sample["bboxes"],
                "cpt_p_id": data_sample["cpt_p_id"],
            }

        return (cpt, image, itm_label, textual_id, cpt_type, cpt_id, img_id)

    def __len__(self) -> int:
        return len(self.captions)

    def _create_caption_list(self, cpt_dict, only_correct_captions=False) -> list:
        caption_list = []
        for i, img_id in enumerate(tqdm(cpt_dict)):
            cpts = cpt_dict.get(img_id).get("captions", cpt_dict.get(img_id))
            for j, cpt_id in enumerate(cpts):
                cpt_data = cpts.get(cpt_id)
                if only_correct_captions and cpt_data.get("label") == 0:
                    continue
                cpt_data["cpt_id"] = cpt_id
                cpt_data["img_id"] = img_id
                caption_list.append(cpt_data)
        return caption_list

    def _create_hnc_ans_label_dicts(self) -> None:
        labels_unique = list(set([cpt.get("textual_label") for cpt in self.captions]))
        ans2label = {}
        label2ans = {}
        for i, label in enumerate(labels_unique):
            ans2label[label] = i
            label2ans[i] = label
        return ans2label, label2ans


def hnc_collate_fn(data):
    captions = []
    images = []
    labels = []
    textual_labels = []
    meta_info = {"cpt_type": [], "cpt_id": [], "img_id": [], "caption": []}
    for _tuple in data:
        captions.append(_tuple[0])
        images.append(_tuple[1])
        labels.append(_tuple[2])
        textual_labels.append(_tuple[3])
        meta_info["cpt_type"].append(_tuple[4])
        meta_info["cpt_id"].append(_tuple[5])
        meta_info["img_id"].append(_tuple[6])
        meta_info["caption"].append(_tuple[0])
    labels = torch.tensor(labels)
    textual_labels = torch.tensor(textual_labels)

    data = {"text": captions, "images": images}
    return data, labels, textual_labels, meta_info
