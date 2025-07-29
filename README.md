# exCLIP
This repository contains the code for the **TMLR'25** paper [*Explaining Caption-Image Interactions in CLIP Models with Second-Order Attributions*](https://openreview.net/forum?id=HUUL19U7HP).

A demo is alreay included in the `demo.ipynb` notebook.

We are still working on cleaning up the code to make it easily accessible and will be updating this repo over the next couple of days.
To stay tuned, we would be glad if you leave a star! 🤩

## Contribution

Our method enables to look into which part of a caption and an image CLIP matches.
We can make arbitrary selections over spans in captions and see which image regions correspond to them or vice versa.
This is demonstrated in the follwing plot.

![example](examples/demo_plot.png)

In the top row, we select spans in captions (yellow) and see what they correspond to in the image above. In the bottom row, we select bounding-boxes in the image (yellow) and see what they correspond to in the caption below. Heatmaps in both images and captions are red for positive and blue for negative values.

For all details, check out the paper!