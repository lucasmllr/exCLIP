import open_clip
from PIL import Image
import torch

from exclip.explanation import OpenClipExplainer
from exclip.tokenization import OpenClipTokenizer


device = 'cuda:2'
model_name = 'ViT-B-16'
pretraining = 'laion2b_s34b_b88k'
model, _, prep = open_clip.create_model_and_transforms(model_name, pretraining)
model.to(device)
model.eval()
explainer = OpenClipExplainer(model, device=device)
tokenizer = OpenClipTokenizer(model_name)

image = Image.open("examples/dogs.jpg")
caption = 'A white husky and a black dog running in a snow covered forest.'
cpt_inpt = tokenizer.tokenize(caption).to(device)
img_inpt = prep(image).unsqueeze(0).to(device)

# computing explanations for all token-patch interactions between the image and caption
interactions = explainer.explain(cpt_inpt, img_inpt, verbose=True)