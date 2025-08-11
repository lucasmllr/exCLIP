import clip
from PIL import Image
from exclip.explanation import OpenAIExplainer
from exclip.tokenization import OpenAITokenizer

device = 'cuda:1'
model, prep = clip.load('ViT-B/16', device=device)
tokenizer = OpenAITokenizer()
explainer = OpenAIExplainer(model, device=device)

image = Image.open("examples/dogs.jpg")
caption = 'A white husky and a black dog running in a snow covered forest.'
cpt_inpt = tokenizer.tokenize(caption).to(device)
img_inpt = prep(image).unsqueeze(0).to(device)

# computing explanations for all token-patch interactions between the image and caption
interactions = explainer.explain(cpt_inpt, img_inpt, verbose=True)