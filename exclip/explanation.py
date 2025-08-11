import math
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from .models.hooks import interpolation_hook, permuting_interpolation_hook


class Explainer(nn.Module):
    def __init__(
        self,
        model: nn.Module,  # pre-trained clip model
        image_dim: int = 224,  # image dimensions (height and width)
        text_seq_len: int = 77,  # max text sequece lenth
        text_ref_len: int = 1,  # fixed token length for text reference
        norm_embeddings: bool = True,  # whether to normalize embeddings to unit length
        scale_cos: bool = False,  # whether to scale cos similarities by a factor of exp(logit_scale)
        device: torch.device = torch.device("cuda:0"),
        itm: bool = False,
        img_ref_type: str = 'zeros'
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.n_train_steps = 0
        self.n_valid_steps = 0
        self.n_test_steps = 0
        self.image_dim = image_dim
        self.text_seq_len = text_seq_len
        self.text_ref_len = text_ref_len
        self.norm_emb = norm_embeddings
        self.scale_cos = scale_cos
        self.attribute = False  # wheather to adjust forward pass for attributions, False for training, if True batch size must be one
        self.counter_powers_of_two = 0
        self.lowest_loss_eval = sys.maxsize
        self.itm = itm
        # refs
        self.txt_ref = self._make_txt_ref()
        self.img_ref_type = img_ref_type
        self.img_ref = self._make_img_ref()

        if self.itm:
            # self.classifier_itm = torch.nn.Sequential(
            #     torch.nn.Linear(1024, 512),
            #     torch.nn.GELU(),
            #     torch.nn.Linear(512, 2),
            # )
            self.classifier_itm = torch.nn.Linear(1, 2)

    def _make_txt_ref(self, text_seq_len=None):
        # clip tokenization uses zero for padding
        if text_seq_len == None:
            text_seq_len = self.text_seq_len
        r = torch.zeros([1, text_seq_len])
        r[0][0] = 49406  # BoS/CLS token
        r[0][self.text_ref_len + 1] = 49407  # EoS token
        return r.long()

    def _make_img_ref(self):
        if self.img_ref_type == 'zeros':
            ref = torch.zeros([1, 3, self.image_dim, self.image_dim])
        elif self.img_ref_type == 'normal':
            ref = torch.randn([1, 3, self.image_dim, self.image_dim])
        return ref

    def legacy_encode_text(self, text: torch.tensor):
        """mostly copied from CLIP.encode_image(), extended by ref shift and attribute option"""
        assert (len(text.shape) == 2), f"expected text to be a (B, S) tensor, but got {text.shape}"
        text = torch.cat(
            [text, self._make_txt_ref(text_seq_len=text.size(1)).to(self.device)]
        )
        x = self.model.token_embedding(text)  #.type(self.model.dtype)
        # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding  #.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)  #.type(self.model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.attribute:  # expand eot idxs to the number of interpolations in the batch
            txt_eot_idx = text[0].argmax(dim=-1)
            ref_eot_idx = text[1].argmax(dim=-1)
            N = x.shape[0] - 1
            eot_idxs = torch.tensor([txt_eot_idx] * N + [ref_eot_idx])
            x = x[torch.arange(x.shape[0]), eot_idxs]
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.model.text_projection
        return x[:-1], x[-1]

    def encode_text(self, text: torch.tensor):
        '''due to different forward passes in the text encoder of OpenAI's and OpenCLIP's impoementation, 
        this method must be implemented in subclasses.'''
        raise NotImplementedError('encode_text must be implemented in subclasses')
    
    def encode_image(self, image: torch.Tensor):
        """adds ref shift to original method"""
        assert (len(image.shape) == 4), f"expected image to be (B, C, D, D) tensor, but got {image.shape}"
        image = torch.cat([image, self.img_ref.to(self.device)])
        # CLIP's encode_image() can be used without adjustment
        x = self.model.encode_image(image)
        return x[:-1], x[-1]

    def logit_cos(self, e_a: torch.Tensor, e_b: torch.Tensor):
        if self.norm_emb:
            e_a = e_a / e_a.norm(dim=1, keepdim=True)
            e_b = e_b / e_b.norm(dim=1, keepdim=True)
        # cosine similarity as logits\
        device = e_a.device
        assert e_b.device == device
        scores = e_a @ e_b.t()
        if self.scale_cos:
            scale = self.model.logit_scale.exp().to(device)
            scores = scale * scores
        return scores, scores.t()

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        img_emb, img_ref = self.encode_image(image)
        txt_emb, txt_ref = self.encode_text(text)
        if self.itm:
            scalars = self.logit_cos(img_emb, txt_emb)[0].diag().unsqueeze(1)
            return self.classifier_itm(scalars)
            # return self.classifier_itm(torch.cat((img_emb, txt_emb), dim=-1))
        return self.logit_cos(img_emb, txt_emb)

    def init_image_attribution(
        self, layer: int, N_interpolations: Union[int, torch.tensor]
    ):
        raise NotImplementedError("init_image_attribution must be implemented in subclasses")

    def init_text_attribution(
        self, layer: int, N_interpolations: Union[int, torch.tensor]
    ):
        raise NotImplementedError("init_text_attribution must be implemented in subclasses")

    def reset_attribution(self):
        self.attribute = False
        if hasattr(self, "txt_hook"):
            self.txt_hook.remove()
            del self.txt_hook
        if hasattr(self, "img_hook"):
            self.img_hook.remove()
            del self.img_hook

    def _compute_integrated_jacobian(
        self,
        embedding: torch.tensor,  # embedding
        features: torch.tensor,  # intermediate / input features
        verbose: bool = True,
    ):
        # TODO: include normalization here
        N, D = embedding.shape
        grads = []
        retain_graph = True
        for d in tqdm(range(D), disable = not verbose):
            if d == D - 1:
                retain_graph = False
            # we can sum gradients over interpolation steps and compute them in a single backward pass
            de_d = torch.autograd.grad(list(embedding[:, d]), features, retain_graph=retain_graph)[0].detach()
            de_d = de_d[:-1].sum(dim=0).cpu()  # integration of grads excluding the reference
            grads.append(de_d)
        J = torch.stack(grads) / N
        return J

    def explain(
        self,
        text: torch.tensor,
        image: torch.tensor,
        text_layer: int = 11,
        image_layer: int = 11,
        N: int = 10,
        cut_txt_padding: bool = True,
        compute_lhs_terms: bool = False,
        verbose: bool = False
    ):
        self.reset_attribution()
        self.init_text_attribution(layer=text_layer, N_interpolations=N)
        self.init_image_attribution(layer=image_layer, N_interpolations=N)

        # embeddings
        txt_emb, txt_ref_emb = self.encode_text(text)
        txt_interm = self.txt_intermediates[0]
        img_emb, img_ref_emb = self.encode_image(image)
        img_interm = self.img_intermediates[0]
        
        # integrated jacobians
        J_txt = self._compute_integrated_jacobian(txt_emb, txt_interm, verbose=verbose)
        J_img = self._compute_integrated_jacobian(img_emb, img_interm, verbose=verbose)
        J_txt = J_txt.to(self.device)
        J_img = J_img.to(self.device)

        # storing intermediate representations and embeddings of inputs and references
        # embeddings for computation of lhs
        ex_txt = txt_emb[0].unsqueeze(0).detach()
        ex_img = img_emb[0].unsqueeze(0).detach()
        er_txt = txt_ref_emb.unsqueeze(0).detach()
        er_img = img_ref_emb.unsqueeze(0).detach()
        # intermediates
        x_txt = txt_interm[0].unsqueeze(0).detach()
        x_img = img_interm[0].unsqueeze(0).detach()
        r_txt = txt_interm[-1].unsqueeze(0).detach()
        r_img = img_interm[-1].unsqueeze(0).detach()

        # deltas for multiplication
        d_txt = x_txt - r_txt
        d_img = x_img - r_img

        # cut text padding, reduces memory consumption
        if cut_txt_padding:
            eot_idx = text.argmax(dim=-1).item()
            J_txt = J_txt[:, : eot_idx + 1, :]
            d_txt = d_txt[:, : eot_idx + 1]

        # text part
        D_emb, S_txt, D_txt = J_txt.shape
        J_txt = J_txt.view((D_emb, S_txt * D_txt)).float()

        # image part
        if hasattr(self.model.visual, "transformer"):  # ViT model
            _, S_img, D_img = J_img.shape
            J_img = J_img.view((D_emb, S_img * D_img)).float()
            d_txt = d_txt.view((S_txt * D_txt, 1)).repeat((1, S_img * D_img))
            d_img = d_img.view((S_img * D_img, 1)).repeat((1, S_txt * D_txt))
        else:  # ResNet model
            _, C_img, D_img_a, D_img_b = J_img.shape
            assert D_img_a == D_img_b
            D_img = D_img_a
            J_img = J_img.view((D_emb, C_img * D_img * D_img)).float()
            d_txt = d_txt.view((S_txt * D_txt, 1)).repeat((1, C_img * D_img * D_img))
            d_img = d_img.view((C_img * D_img * D_img, 1)).repeat((1, S_txt * D_txt))

        # multiplication
        J = torch.mm(J_txt.T, J_img)
        A = d_txt * J * d_img.T

        # normalizing attributions
        ex_img_norm = torch.norm(ex_img)
        ex_txt_norm = torch.norm(ex_txt)
        ex_img_norm, ex_txt_norm = ex_img_norm.cpu(), ex_txt_norm.cpu()
        A = A / ex_img_norm / ex_txt_norm
        
        # scaling attributions
        if self.scale_cos:
            scale = self.model.logit_scale.exp()
            A = A * scale

        # collapsing embedding dimensions and reshaping attributions
        if hasattr(self.model.visual, "transformer"):  # ViT model
            A = A.view((S_txt, D_txt, S_img, D_img))
            A = A.sum(dim=(1, 3))
            n_patches = int(math.sqrt(A.shape[1] - 1))  # removing cls patch
            n_tokens = A.shape[0]
            A = A[:, 1:].view(n_tokens, n_patches, n_patches)
        else:  # ResNet model
            A = A.view((S_txt, D_txt, C_img, D_img, D_img))
            A = A.sum(dim=(1, 2))

        A = A.detach().cpu()

        if compute_lhs_terms:
            score = self.logit_cos(ex_txt.float(), ex_img.float())[0].item()
            txt_ref_sim = self.logit_cos(ex_txt.float(), er_img.float())[0].item()
            img_ref_sim = self.logit_cos(er_txt.float(), ex_img.float())[0].item()
            ref_ref_sim = self.logit_cos(er_txt.float(), er_img.float())[0].item()
            return A, score, txt_ref_sim, img_ref_sim, ref_ref_sim
        else:
            return A


class OpenAIExplainer(Explainer):

    def init_image_attribution(self, layer: int, N_interpolations: Union[int, torch.tensor]):
        self.img_intermediates = []
        if hasattr(self.model.visual, "transformer"):  # ViT model
            assert layer < len(
                self.model.visual.transformer.resblocks
            ), f"There is no layer {layer} in the vision model."
            self.img_hook = self.model.visual.transformer.resblocks[
                layer
            ].register_forward_pre_hook(
                permuting_interpolation_hook(
                    N_interpolations, cache=self.img_intermediates
                )
                # saving_hook(self.img_intermediates)
            )
        else:  # ResNet model
            assert layer <= 4, f"There is no layer {layer} in the vision model."
            res_layer = eval(f"self.model.visual.layer{layer}")
            self.img_hook = res_layer.register_forward_pre_hook(
                interpolation_hook(N_interpolations, cache=self.img_intermediates)
                # saving_hook(self.img_intermediates)
            )

    def init_text_attribution(self, layer: int, N_interpolations: Union[int, torch.tensor]):
        assert layer < len(
            self.model.transformer.resblocks
        ), f"There is no layer {layer} in the text model."
        self.txt_intermediates = []
        self.txt_hook = self.model.transformer.resblocks[
            layer
        ].register_forward_pre_hook(
            permuting_interpolation_hook(
                N_interpolations, cache=self.txt_intermediates
            )
            # saving_hook(self.txt_intermediates)
        )

    def encode_text(self, text):
        """mostly copied from CLIP.encode_image(), but pooling needs to be adjusted to the number of interpolations N"""
        # attaching reference to the batch
        text = torch.cat([text, self.txt_ref.to(self.device)])
        
        # copied from original implementation
        x = self.model.token_embedding(text).type(self.model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # expanding eot pooling to N interpolation steps
        txt_eot_idx = text[0].argmax(dim=-1)
        ref_eot_idx = text[1].argmax(dim=-1)
        N = x.shape[0] - 1
        eot_idxs = torch.tensor([txt_eot_idx] * N + [ref_eot_idx])
        x = x[torch.arange(x.shape[0]), eot_idxs]

        # final projection
        x = x @ self.model.text_projection

        # separating interpolations and reference
        return x[:-1], x[-1]


class OpenClipExplainer(Explainer):

    def init_image_attribution(
        self, layer: int, N_interpolations: Union[int, torch.tensor]
    ):
        self.img_intermediates = []
        assert layer < len(self.model.visual.transformer.resblocks), f"There is no layer {layer} in the vision model."
        self.img_hook = self.model.visual.transformer.resblocks[
            layer].register_forward_pre_hook(
                interpolation_hook(
                    N_interpolations, cache=self.img_intermediates
                )
            )

    def init_text_attribution(
        self, layer: int, N_interpolations: Union[int, torch.tensor]
    ):
        assert layer < len(self.model.transformer.resblocks), f"There is no layer {layer} in the text model."
        self.txt_intermediates = []
        self.txt_hook = self.model.transformer.resblocks[layer].register_forward_pre_hook(
            interpolation_hook(
                N_interpolations, cache=self.txt_intermediates
            )
        )

    def encode_text(self, text):
        """mostly copied from CLIP.encode_image(), but pooling needs to be adjusted to the number of interpolations N"""
        # attaching reference to the batch
        text = torch.cat([text, self.txt_ref.to(self.device)])
        
        # copied from original implementation
        cast_dtype = self.model.transformer.get_cast_dtype()
        x = self.model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding.to(cast_dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        
        # x = text_global_pool(x, text, self.text_pool_type)
        # expanding eot pooling to N interpolation steps
        txt_eot_idx = text[0].argmax(dim=-1)
        ref_eot_idx = text[1].argmax(dim=-1)
        N = x.shape[0] - 1
        eot_idxs = torch.tensor([txt_eot_idx] * N + [ref_eot_idx])
        x = x[torch.arange(x.shape[0]), eot_idxs]

        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                x = self.model.text_projection(x)
            else:
                x = x @ self.model.text_projection

        return x[:-1], x[-1]
    
    
if __name__ == "__main__":

    import clip
    from PIL import Image
    from .models.tokenization import ClipTokenizer

    model_name = "ViT-B/16"
    print("init model")
    device = torch.device("cuda:3")
    model, prep = clip.load(model_name, device=device)
    xclip = Explainer(
        model,
        device=device,
        img_ref_type='normal',
        text_ref_len=0
    )
    tokenizer = ClipTokenizer()

    print("prep input")
    image = prep(Image.open("examples/dogs.jpg")).unsqueeze(0).to(device)
    caption = tokenizer.tokenize('Two dogs running in the snow.').to(device)

    clip_txt_emb = model.encode_text(caption)
    clip_img_emb = model.encode_image(image)
    clip_sim = torch.nn.functional.cosine_similarity(clip_img_emb, clip_txt_emb)

    xclip_txt_emb = xclip.encode_text(caption)[0]
    xclip_img_emb = xclip.encode_image(image)[0]
    xclip_sim = torch.nn.functional.cosine_similarity(xclip_img_emb, xclip_txt_emb)

    # A, s, rtxt, rimg, rr = xclip.explain(
    #     text=caption,
    #     image=image,
    #     text_layer=11,
    #     image_layer=11,
    #     N=5,
    #     clip_txt_padding=True,
    #     compute_lhs_terms=True, 
    #     verbose=True
    # )

    Ab, sb, rtxtb, rimgb, rrb = xclip.explain_no_batch(
        text=caption,
        image=image,
        text_layer=11,
        image_layer=11,
        N=5,
        cut_txt_padding=True,
        compute_lhs_terms=True, 
        verbose=True
    )
    
    breakpoint()
