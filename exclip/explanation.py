import math
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


# hooks for the computation of interpolations

def interpolate_reference_embedding(e: Tensor, interpolations: Union[int, Tensor]):
    """Linearly interpolates N steps between the instance at index=1 and the reference at index=2.
    The batch must consist only of instance and reference."""
    if e.shape[0] != 2:
        breakpoint()
    assert e.shape[0] == 2
    d = len(e.shape)
    device = e.device
    dtype = e.dtype
    if isinstance(interpolations, int):
        assert interpolations > 0
        s = 1 / interpolations
        a = torch.arange(1, 0, -s, dtype=dtype).to(device)
    elif isinstance(interpolations, Tensor):
        assert len(interpolations.shape) == 1
        a = interpolations.clone().type(dtype).to(device)
    else:
        raise TypeError(
            f"interpolations must be int or tensor, got {type(interpolations)}"
        )
    x, r = e[0].unsqueeze(0), e[-1].unsqueeze(0)
    for _ in range(d - 1):  # same dims as e
        a.unsqueeze_(1)
    g = r + a * (x - r)
    g = torch.cat([g, r])
    return g


def interpolation_hook(interpolations: Union[int, Tensor], cache: list):
    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], interpolations)
        cache.append(g)
        return (g,) + inpt[1:]

    return hook


def permuting_interpolation_hook(interpolations: Union[int, Tensor], cache: list):
    def hook(model, inpt):
        e = inpt[0]
        # batch shape: from (S, B, D) to (B, S, D)
        e = e.permute(1, 0, 2)
        g = interpolate_reference_embedding(e, interpolations)
        cache.append(g)
        # back to batch shape (S, B, D)
        g = g.permute(1, 0, 2)
        return (g,) + inpt[1:]

    return hook


# abstract explainer class

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
        # refs
        self.txt_ref = self._make_txt_ref()
        self.img_ref_type = img_ref_type
        self.img_ref = self._make_img_ref()

    def _make_txt_ref(self, text_seq_len=None):
        '''Creates the text reference.'''
        # clip tokenization uses zero for padding
        if text_seq_len == None:
            text_seq_len = self.text_seq_len
        r = torch.zeros([1, text_seq_len])
        r[0][0] = 49406  # BoS/CLS token
        r[0][self.text_ref_len + 1] = 49407  # EoS token
        return r.long()

    def _make_img_ref(self):
        '''Creates the image reference.'''
        if self.img_ref_type == 'zeros':
            ref = torch.zeros([1, 3, self.image_dim, self.image_dim])
        elif self.img_ref_type == 'normal':
            ref = torch.randn([1, 3, self.image_dim, self.image_dim])
        return ref

    def encode_text(self, text: torch.tensor):
        '''due to different forward passes in the text encoders of OpenAI's and OpenCLIP's implementation, 
        this method must be implemented in subclasses.'''
        raise NotImplementedError('encode_text must be implemented in subclasses')
    
    def encode_image(self, image: torch.Tensor):
        '''adds the image reference to to original original batch which can be wrapped otherwise'''
        assert (len(image.shape) == 4), f"expected image to be (B, C, D, D) tensor, but got {image.shape}"
        image = torch.cat([image, self.img_ref.to(self.device)])
        x = self.model.encode_image(image)
        return x[:-1], x[-1]

    def logit_cos(self, e_a: torch.Tensor, e_b: torch.Tensor):
        '''Computes cosine similarity between two embeddings, optionally scaled by a learned factor.'''
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
        '''Forward pass through the model, computing cosine similarity between image and text embeddings.'''
        img_emb, img_ref = self.encode_image(image)
        txt_emb, txt_ref = self.encode_text(text)
        return self.logit_cos(img_emb, txt_emb)

    def init_image_attribution(self, layer: int, N_interpolations: Union[int, torch.tensor]):
        '''Due to differences between OpenAI's and OpenCLIP's implementations, the initialization of the attribution
        computation and the involved registration of hooks must be implemented separately in subclasses.'''
        raise NotImplementedError("init_image_attribution must be implemented in subclasses")

    def init_text_attribution(self, layer: int, N_interpolations: Union[int, torch.tensor]):
        '''Due to differences between OpenAI's and OpenCLIP's implementations, the initialization of the attribution
        computation and the involved registration of hooks must be implemented separately in subclasses.'''
        raise NotImplementedError("init_text_attribution must be implemented in subclasses")

    def reset_attribution(self):
        '''Resets the attribution computation by removing all hooks and clearing cached intermediate representations.'''
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
        '''Computes integrated Jacobian for the given embedding w.r.t features (cf. Equation 9 in the paper).'''
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
        '''Computes our second-order explanations for all token-patch interactions between the image and 
        caption (cf. Equation 10 in the paper).'''
        
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


# OpenAi and OpenClip Explaners

class OpenAIExplainer(Explainer):

    def init_image_attribution(self, layer: int, N_interpolations: Union[int, torch.tensor]):
        '''Initializes the image attribution computation by registering a forward hook to a defined layer
        that computes interpolations between the actual input image representation and the image referece.'''
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
        '''Initializes the text attribution computation by registering a forward hook to a defined layer
        that computes interpolations between the actual captio input and referece.'''
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
        """Forward pass through the text encoder.
        Mostly copied from CLIP.encode_image(), but the eos pooling needs to be adjusted to the number of interpolations N"""
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

    def init_image_attribution(self, layer: int, N_interpolations: Union[int, torch.tensor]):
        '''Initializes the image attribution computation by registering a forward hook to a defined layer
        that computes interpolations between the actual input image representation and the image referece.'''
        self.img_intermediates = []
        assert layer < len(self.model.visual.transformer.resblocks), f"There is no layer {layer} in the vision model."
        self.img_hook = self.model.visual.transformer.resblocks[
            layer].register_forward_pre_hook(
                interpolation_hook(
                    N_interpolations, cache=self.img_intermediates
                )
            )

    def init_text_attribution(self, layer: int, N_interpolations: Union[int, torch.tensor]):
        '''Initializes the text attribution computation by registering a forward hook to a defined layer
        that computes interpolations between the actual captio input and referece.'''
        assert layer < len(self.model.transformer.resblocks), f"There is no layer {layer} in the text model."
        self.txt_intermediates = []
        self.txt_hook = self.model.transformer.resblocks[layer].register_forward_pre_hook(
            interpolation_hook(
                N_interpolations, cache=self.txt_intermediates
            )
        )

    def encode_text(self, text):
        """Forward pass through the text encoder. Mostly copied from the OpenClip implementation, 
        but the eos pooling needs to be adjusted to the number of interpolations N"""
        
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
    