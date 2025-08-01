from typing import Any, Tuple

import torchvision.datasets as dset
from pycocotools.coco import COCO


class CocoDataset(dset.CocoDetection):
    def __init__(
        self, root: str, cpts_ann: str, instances_ann: str, load_instances: bool = False
    ) -> None:
        super().__init__(root, cpts_ann)
        self.load_instances = load_instances
        self.ids = list(sorted(self.coco.anns.keys()))
        self.instances = COCO(instances_ann)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        img_id = self.coco.anns[id]["image_id"]
        image = self._load_image(img_id)
        caption = self.coco.anns[id]["caption"]

        if self.transforms is not None:
            image, caption = self.transforms(image, caption)

        instance_ids = self.instances.getAnnIds(img_id)
        instance_ann = self.instances.loadAnns(instance_ids)
        for inst in instance_ann:
            inst_id = inst["category_id"]
            inst_name = self.instances.loadCats(inst_id)[0]["name"]
            inst["category"] = inst_name
        return {
            "image": image,
            "text": caption,
            "annotation": instance_ann,
            "image_id": img_id,
        }


def coco_collate_fn(data):
    captions = []
    images = []
    image_ids = []
    annotations = []
    for _tuple in data:
        captions.append(_tuple["text"])
        images.append(_tuple["image"])
        image_ids.append(_tuple["image_id"])
        annotations.append(_tuple["annotation"])
    data = {
        "text": captions,
        "images": images,
        "image_ids": image_ids,
        "annotations": annotations,
    }
    return data
