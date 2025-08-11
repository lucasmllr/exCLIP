# exCLIP
This repository contains the code for the **TMLR'25** paper [*Explaining Caption-Image Interactions in CLIP Models with Second-Order Attributions*](https://openreview.net/forum?id=HUUL19U7HP).

We are still working on cleaning up the code to make it easily accessible and will be updating this repo over the next couple of days.
To stay tuned, we would be glad if you leave a star! 🤩

## Contribution

Our method enables to look into which part of a caption and an image CLIP matches.
We can make arbitrary selections over spans in captions and see which image regions correspond to them or vice versa.
This is demonstrated in the follwing plot.

![example](examples/demo_plot.png)

In the top row, we select spans in captions (yellow) and see what they correspond to in the image above. In the bottom row, we select bounding-boxes in the image (yellow) and see what they correspond to in the caption below. Heatmaps in both images and captions are red for positive and blue for negative values.

For all details, check out the paper!

## Installation

To use our `exclip` package, simply install it with (The pip version is currently behind, will be updated soon):
```bash
$ pip install exclip
```
You also need to install OpenAI's [clip](https://github.com/openai/CLIP) package with the following command (since it is not available on PyPI):
```bash
$ pip install git+https://github.com/openai/CLIP.git
```

Alternatively, you can directly install this repository:
```bash
$ pip install git+https://github.com/lucasmllr/exCLIP
```
or clone it and run `$ pip install .` inside the cloned directory.
The latter two versions already include the clip installation, too.

## Getting started
The following minimal example initializes a clip model, wraps it into our Explainer and computes interaction explanations for a given image-caption pair.
```python
import clip
from PIL import Image
from exclip import Explainer
from exclip.models.tokenization import ClipTokenizer

device = 'cuda:1'
model, prep = clip.load('ViT-B/16', device=device)
tokenizer = ClipTokenizer()
explainer = Explainer(model, device=device)

image = Image.open("examples/dogs.jpg")
caption = 'A white husky and a black dog running in a snow covered forest.'
txt_inpt = tokenizer.tokenize(caption).to(device)
img_inpt = prep(image).unsqueeze(0).to(device)

# computing explanations for all token-patch interactions between the image and caption
interactions = explainer.explain(txt_inpt, img_inpt)
```

This example is also inclued in `minimal_example_openai.py` and `minimap_example_open_clip.py` contains an equivalent for an OpenClip model. 

The `demo.ipynb` notebook also shows how to visualize the resulting explanations for different token ranges in the caption.