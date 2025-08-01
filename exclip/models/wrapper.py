import math
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from ..utils.matmul import split_mm
from .hooks import interpolation_hook, transformer_interpolation_hook


class exCLIP(nn.Module):
    def __init__(
        self,
        model: nn.Module,  # pre-trained clip model
        image_dim: int = 224,  # image dimensions (height and width)
        text_seq_len: int = 77,  # max text sequece lenth
        text_ref_len: int = 3,  # fixed token length for text reference
        shift_embeddings: bool = True,  # whether to shift embeddings by the reference
        norm_embeddings: bool = True,  # whether to normalize embeddings to unit length
        scale_cos: bool = True,  # whether to scale cos similarities by a factor of exp(logit_scale)
        device: torch.device = torch.device("cuda:0"),
        itm: bool = False,
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
        self.shift_emb = shift_embeddings
        self.norm_emb = norm_embeddings
        self.scale_cos = scale_cos
        self.txt_ref = self._make_txt_ref()
        self.img_ref = self._make_img_ref()
        self.attribute = False  # wheather to adjust forward pass for attributions, False for training, if True batch size must be one
        self.counter_powers_of_two = 0
        self.lowest_loss_eval = sys.maxsize
        self.itm = itm

        if self.itm:
            self.classifier_itm = torch.nn.Linear(1, 2)

    def _make_txt_ref(self, text_seq_len=None):
        # clip tokenization uses zero for padding
        if text_seq_len == None:
            text_seq_len = self.text_seq_len
        r = torch.zeros([1, text_seq_len])
        r[0][0] = 49406  # BoS/CLS token
        r[0][self.text_ref_len] = 49407  # EoS token
        return r.long()

    def _make_img_ref(self):
        return torch.zeros([1, 3, self.image_dim, self.image_dim])

    def encode_text(self, text: torch.tensor):
        """mostly copied from CLIP.encode_image(), extended by ref shift and attribute option"""
        assert len(text.shape) == 2, (
            f"expected text to be a (B, S) tensor, but got {text.shape}"
        )
        text = torch.cat(
            [text, self._make_txt_ref(text_seq_len=text.size(1)).to(self.device)]
        )
        x = self.model.token_embedding(text).type(self.model.dtype)
        # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if (
            self.attribute
        ):  # expand eot idxs to the number of interpolations in the batch
            txt_eot_idx = text[0].argmax(dim=-1)
            ref_eot_idx = text[1].argmax(dim=-1)
            N = x.shape[0] - 1
            eot_idxs = torch.tensor([txt_eot_idx] * N + [ref_eot_idx])
            x = x[torch.arange(x.shape[0]), eot_idxs]
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.model.text_projection
        if self.shift_emb:
            x = x - x[-1]
        return x[:-1], x[-1]

    def encode_image(self, image: torch.Tensor):
        """adds ref shift to original method"""
        assert len(image.shape) == 4, (
            f"expected image to be (B, C, D, D) tensor, but got {image.shape}"
        )
        image = torch.cat([image, self.img_ref.to(self.device)])
        # CLIP's encode_image() can be used without adjustment
        x = self.model.encode_image(image)
        if self.shift_emb:
            x = x - x[-1]
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
        self.img_intermediates = []
        if hasattr(self.model.visual, "transformer"):  # ViT model
            assert layer < len(self.model.visual.transformer.resblocks), (
                f"There is no layer {layer} in the vision model."
            )
            self.img_hook = self.model.visual.transformer.resblocks[
                layer
            ].register_forward_pre_hook(
                transformer_interpolation_hook(
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

    def init_text_attribution(
        self, layer: int, N_interpolations: Union[int, torch.tensor]
    ):
        assert layer < len(self.model.transformer.resblocks), (
            f"There is no layer {layer} in the text model."
        )
        self.txt_intermediates = []
        self.txt_hook = self.model.transformer.resblocks[
            layer
        ].register_forward_pre_hook(
            transformer_interpolation_hook(
                N_interpolations, cache=self.txt_intermediates
            )
            # saving_hook(self.txt_intermediates)
        )
        self.attribute = True

    def reset_attribution(self):
        self.attribute = False
        if hasattr(self, "txt_hook"):
            self.txt_hook.remove()
            del self.txt_hook
        if hasattr(self, "img_hook"):
            self.img_hook.remove()
            del self.img_hook

    def _compute_jacobians(
        self,
        e: torch.tensor,  # embedding
        x: torch.tensor,  # intermediate / input features
        verbose: bool = True,
    ):
        D = e.shape[1]
        grads = []
        retain_graph = True
        for d in tqdm(range(D), disable=not verbose):
            if d == D - 1:
                retain_graph = False
            # we can sum gradients and compute them in a single backward pass
            de_d = torch.autograd.grad(list(e[:, d]), x, retain_graph=retain_graph)[
                0
            ].detach()
            de_d = de_d[:-1].sum(dim=0).cpu()  # integration of grads excluding ref
            # de_d = de_d.sum(dim=0).cpu()
            grads.append(de_d)
        J = torch.stack(grads)
        return J

    def attribute_prediction(
        self,
        text: torch.tensor,
        image: torch.tensor,
        text_layer: int,
        image_layer: int,
        N: int = 100,
        batch_size: Optional[int] = None,
        compress_emb_dims: bool = True,
        clip_txt_padding: bool = True,
        compute_lhs_terms: bool = False,
        verbose: bool = False,
    ):
        if batch_size is None:
            batch_size = N
        n_batches = math.ceil(N / batch_size)
        s = 1 / N
        a = torch.arange(1, 0, -s).to(self.device)
        jacobians_txt = []
        jacobians_img = []

        for b in range(n_batches):
            # print(f'\nBatch: {b}/{n_batches}')
            first = b * batch_size
            last = (b + 1) * batch_size
            if last < N:
                a_b = a[first:last]
            else:
                a_b = a[first:]
            self.init_text_attribution(layer=text_layer, N_interpolations=a_b)
            txt_emb, txt_ref_emb = self.encode_text(text)
            txt_interm = self.txt_intermediates[0]
            jb_txt = self._compute_jacobians(txt_emb, txt_interm, verbose=verbose)
            jacobians_txt.append(jb_txt)
            self.init_image_attribution(layer=image_layer, N_interpolations=a_b)
            img_emb, img_ref_emb = self.encode_image(image)
            img_interm = self.img_intermediates[0]
            jb_img = self._compute_jacobians(img_emb, img_interm, verbose=verbose)
            jacobians_img.append(jb_img)

            # storing intermediate representations and embeddings of inputs and references
            if b == 0:
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

            self.reset_attribution()

        J_txt = torch.stack(jacobians_txt).sum(dim=0) / N  # integration
        J_img = torch.stack(jacobians_img).sum(dim=0) / N  # integration

        J_txt = J_txt.to(self.device)
        J_img = J_img.to(self.device)

        d_txt = x_txt - r_txt
        d_img = x_img - r_img

        if clip_txt_padding:
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

        # NOTE: when clipped caption-image attributions fit into gpu for short captions
        # J = split_mm(J_txt.T, J_img, splits=2, device=self.device)
        J = torch.mm(J_txt.T, J_img)
        # d_txt, J, d_img = d_txt.cpu(), J.cpu(), d_img.cpu()
        A = d_txt * J * d_img.T
        scale = self.model.logit_scale.exp()
        ex_img_norm = torch.norm(ex_img)
        ex_txt_norm = torch.norm(ex_txt)
        # ex_img_norm, ex_txt_norm = ex_img_norm.cpu(), ex_txt_norm.cpu()
        A = A / ex_img_norm / ex_txt_norm  # * scale
        # A = A / ex_img_norm / ex_txt_norm

        if hasattr(self.model.visual, "transformer"):  # ViT model
            A = A.view((S_txt, D_txt, S_img, D_img))
            if compress_emb_dims:
                A = A.sum(dim=(1, 3))
        else:  # ResNet model
            A = A.view((S_txt, D_txt, C_img, D_img, D_img))
            if compress_emb_dims:
                A = A.sum(dim=(1, 2))

        if compute_lhs_terms:
            score = self.logit_cos(ex_txt.float(), ex_img.float())[0].item()
            txt_ref_sim = self.logit_cos(ex_txt.float(), er_img.float())[0].item()
            img_ref_sim = self.logit_cos(er_txt.float(), ex_img.float())[0].item()
            ref_ref_sim = self.logit_cos(er_txt.float(), er_img.float())[0].item()
            return A, score, txt_ref_sim, img_ref_sim, ref_ref_sim
        else:
            return A

    def minimal_attr(
        self,
        text: torch.tensor,
        image: torch.tensor,
        text_layer: int,
        image_layer: int,
        N: int = 100,
        clip_txt_padding: bool = True,
        verbose: bool = True,
        compute_lhs_terms: bool = True,
    ):
        # initialization
        self.init_text_attribution(layer=text_layer, N_interpolations=N)
        self.init_image_attribution(layer=image_layer, N_interpolations=N)

        # compunting embeddings and jacobians for interpolations
        # NOTE: embedding computation has no normalization
        # text side
        txt_emb, txt_ref_emb = self.encode_text(text)
        txt_interm = self.txt_intermediates[0]
        # txt_emb = txt_emb / torch.norm(txt_emb, dim=1, keepdim=True)
        j_txt = self._compute_jacobians(txt_emb, txt_interm, verbose=verbose)
        # image side
        img_emb, img_ref_emb = self.encode_image(image)
        img_interm = self.img_intermediates[0]
        # img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
        j_img = self._compute_jacobians(img_emb, img_interm, verbose=verbose)

        # embeddings of original input and reference
        ex_txt = txt_emb[0].unsqueeze(0).detach()
        ex_img = img_emb[0].unsqueeze(0).detach()
        er_txt = txt_ref_emb.unsqueeze(0).detach()
        er_img = img_ref_emb.unsqueeze(0).detach()
        # input/intermediate representations of input and reference
        x_txt = txt_interm[0].unsqueeze(0).detach()
        x_img = img_interm[0].unsqueeze(0).detach()
        r_txt = txt_interm[-1].unsqueeze(0).detach()
        r_img = img_interm[-1].unsqueeze(0).detach()

        # deltas for multiplication
        d_txt = x_txt - r_txt
        d_img = x_img - r_img

        # normalizing integrated jacobians
        J_txt = (j_txt / N).to(self.device)
        J_img = (j_img / N).to(self.device)

        # clip text padding
        if clip_txt_padding:
            eot_idx = text.argmax(dim=-1).item()
            J_txt = J_txt[:, : eot_idx + 1, :]
            d_txt = d_txt[:, : eot_idx + 1]

        # reshaping for multiplication
        D_emb, S_txt, D_txt = J_txt.shape
        _, S_img, D_img = J_img.shape
        J_txt = J_txt.view((D_emb, S_txt * D_txt)).float()
        J_img = J_img.view((D_emb, S_img * D_img)).float()
        d_txt = d_txt.view((S_txt * D_txt)).unsqueeze(1).repeat((1, S_img * D_img))
        d_img = d_img.view((S_img * D_img)).unsqueeze(1).repeat((1, S_txt * D_txt))

        # multiplication
        # J_txt, J_img = J_txt.cpu(), J_img.cpu()
        # d_txt, d_img = d_txt.cpu(), d_img.cpu()
        # ex_txt, ex_img = ex_txt.cpu(), ex_img.cpu()
        # er_txt, er_img = er_txt.cpu(), er_img.cpu()
        J = torch.mm(J_txt.T, J_img)
        A = d_txt * J * d_img.T
        # scaling attributions
        scale = self.model.logit_scale.exp()
        ex_img_norm = torch.norm(ex_img)
        ex_txt_norm = torch.norm(ex_txt)
        A = A / ex_img_norm / ex_txt_norm  # * scale
        A = A.view((S_txt, D_txt, S_img, D_img))
        A = A.sum(dim=(1, 3))

        if compute_lhs_terms:
            # left-hand side terms
            # NOTE: logit_cos() has normalization
            score = self.logit_cos(ex_txt.float(), ex_img.float())[0].item()
            txt_ref_sim = self.logit_cos(ex_txt.float(), er_img.float())[0].item()
            img_ref_sim = self.logit_cos(er_txt.float(), ex_img.float())[0].item()
            ref_ref_sim = self.logit_cos(er_txt.float(), er_img.float())[0].item()
            A_tot = A.sum().item()
            return A, score, txt_ref_sim, img_ref_sim, ref_ref_sim
        else:
            return A

    def Jacobians_x_embeddings(
        self,
        text: torch.tensor,
        image: torch.tensor,
        text_layer: int,
        image_layer: int,
        clip_txt_padding: bool = True,
        verbose: bool = True,
    ):
        # initialization
        self.init_text_attribution(layer=text_layer, N_interpolations=1)
        self.init_image_attribution(layer=image_layer, N_interpolations=1)

        # compunting embeddings and jacobians for interpolations
        # NOTE: embedding computation has no normalization
        # text side
        txt_emb, _ = self.encode_text(text)
        txt_interm = self.txt_intermediates[0]
        # txt_emb = txt_emb / torch.norm(txt_emb, dim=1, keepdim=True)
        j_txt = self._compute_jacobians(txt_emb, txt_interm, verbose=verbose)
        # image side
        img_emb, _ = self.encode_image(image)
        img_interm = self.img_intermediates[0]
        # img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
        j_img = self._compute_jacobians(img_emb, img_interm, verbose=verbose)

        # embeddings of original input and reference
        ex_txt = txt_emb[0].unsqueeze(0).detach()
        ex_img = img_emb[0].unsqueeze(0).detach()
        # input / intermediate representations of input and reference
        x_txt = txt_interm[0].unsqueeze(0).detach()
        x_img = img_interm[0].unsqueeze(0).detach()

        # jacobians
        J_txt = j_txt.to(self.device)
        J_img = j_img.to(self.device)

        # clip text padding
        if clip_txt_padding:
            eot_idx = text.argmax(dim=-1).item()
            J_txt = J_txt[:, : eot_idx + 1, :]
            x_txt = x_txt[:, : eot_idx + 1]

        # reshaping for multiplication
        D_emb, S_txt, D_txt = J_txt.shape
        _, S_img, D_img = J_img.shape
        J_txt = J_txt.view((D_emb, S_txt * D_txt)).float()
        J_img = J_img.view((D_emb, S_img * D_img)).float()
        x_txt = x_txt.view((S_txt * D_txt)).unsqueeze(1).repeat((1, S_img * D_img))
        x_img = x_img.view((S_img * D_img)).unsqueeze(1).repeat((1, S_txt * D_txt))

        # multiplication
        J = torch.mm(J_txt.T, J_img)
        A = x_txt * J * x_img.T
        # scaling attributions
        # scale = self.model.logit_scale.exp()
        ex_img_norm = torch.norm(ex_img)
        ex_txt_norm = torch.norm(ex_txt)
        # A = A / ex_img_norm / ex_txt_norm  # * scale
        A = A.view((S_txt, D_txt, S_img, D_img))
        A = A.sum(dim=(1, 3))

        score = self.logit_cos(ex_txt.float(), ex_img.float())[0].item()

        return A, score

    def multiply(
        self,
        text: torch.tensor,
        image: torch.tensor,
        text_layer: int,
        image_layer: int,
        clip_txt_padding: bool = True,
    ):
        # initialization
        self.init_text_attribution(layer=text_layer, N_interpolations=1)
        self.init_image_attribution(layer=image_layer, N_interpolations=1)

        # text side
        txt_emb, _ = self.encode_text(
            text
        )  # output not needed, only interm saved by hook
        txt_interm = self.txt_intermediates[0][0:1]  # only input, no ref
        if clip_txt_padding:
            eot_idx = text.argmax(dim=-1).item()
            txt_interm = txt_interm[:, : eot_idx + 1, :]
        e_txt = self.model.ln_final(txt_interm)  # .type(self.model.dtype)
        # image side
        img_emb, _ = self.encode_image(
            image
        )  # output not needed, only interm saved by hook
        img_interm = self.img_intermediates[0][0:1]  # only input, no ref
        e_img = img_interm @ self.model.visual.proj

        A = e_txt[0] @ e_img[0].T

        ex_txt = txt_emb[0].unsqueeze(0).detach()
        ex_img = img_emb[0].unsqueeze(0).detach()
        score = self.logit_cos(ex_txt.float(), ex_img.float())[0].item()

        return A, score

    def itsm(
        self, text: torch.tensor, image: torch.tensor, clip_txt_padding: bool = True
    ):
        # TODO: implement nicer with a hook
        # final sequential text embeddings
        x_txt = self.model.token_embedding(text).type(
            self.model.dtype
        )  # [batch_size, n_ctx, d_model]
        x_txt = x_txt + self.model.positional_embedding.type(self.model.dtype)
        x_txt = x_txt.permute(1, 0, 2)  # NLD -> LND
        x_txt = self.model.transformer(x_txt)
        x_txt = x_txt.permute(1, 0, 2)  # LND -> NLD
        if clip_txt_padding:
            eot_idx = text.argmax(dim=-1).item()
            x_txt = x_txt[:, : eot_idx + 1, :]
        x_txt = self.model.ln_final(x_txt)  # .type(self.model.dtype)

        e_txt = x_txt @ self.model.text_projection
        e_txt = e_txt[0]

        # final image patch embeddings
        x_img = self.model.visual.conv1(
            image.type(self.model.dtype)
        )  # shape = [*, width, grid, grid]
        x_img = x_img.reshape(
            x_img.shape[0], x_img.shape[1], -1
        )  # shape = [*, width, grid ** 2]
        x_img = x_img.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_img = torch.cat(
            [
                self.model.visual.class_embedding.to(x_img.dtype)
                + torch.zeros(
                    x_img.shape[0],
                    1,
                    x_img.shape[-1],
                    dtype=x_img.dtype,
                    device=x_img.device,
                ),
                x_img,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x_img = x_img + self.model.visual.positional_embedding.to(x_img.dtype)
        x_img = self.model.visual.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)  # NLD -> LND
        x_img = self.model.visual.transformer(x_img)
        x_img = x_img.permute(1, 0, 2)  # LND -> NLD
        e_img = x_img @ self.model.visual.proj
        e_img = e_img[0]

        A = e_txt @ e_img.transpose(0, 1)

        # second forward pass to compute score..
        txt_emb, _ = self.encode_text(text)
        img_emb, _ = self.encode_image(image)
        ex_txt = txt_emb[0].unsqueeze(0).detach()
        ex_img = img_emb[0].unsqueeze(0).detach()
        score = self.logit_cos(ex_txt.float(), ex_img.float())[0].item()

        return A, score

    def img_img_attr(
        self,
        image_a: torch.tensor,
        image_b: torch.tensor,
        image_layer: int = 11,
        N: int = 50,
        verbose: bool = True,
    ):
        self.init_image_attribution(layer=image_layer, N_interpolations=N)
        emb_a = self.encode_image(image_a)
        ipt_a = self.img_intermediates[0]
        emb_b = self.encode_image(image_b)
        ipt_b = self.img_intermediates[1]
        A, score, ra, rb, rr = self.attr_computation(
            emb_a, emb_b, ipt_a, ipt_b, N_interpolations=N
        )
        A = A.sum(dim=(1, 3))  # summing out emb dims
        p = int(math.sqrt(A.shape[0] - 1))  # number of patches
        A = A[1:, 1:].view((p, p, p, p))
        return A, score, ra, rb, rr

    def txt_txt_attr(
        self,
        text_a: torch.tensor,
        text_b: torch.tensor,
        text_layer: int = 11,
        N: int = 50,
        verbose: bool = True,
    ):
        self.init_text_attribution(layer=text_layer, N_interpolations=N)
        emb_a = self.encode_text(text_a)
        ipt_a = self.txt_intermediates[0]
        emb_b = self.encode_text(text_b)
        ipt_b = self.txt_intermediates[1]
        A, score, ra, rb, rr = self.attr_computation(
            emb_a, emb_b, ipt_a, ipt_b, N_interpolations=N
        )
        A = A.sum(dim=(1, 3))  # summing out emb dims
        return A, score, ra, rb, rr

    def attr_computation(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        ipt_a: Tensor,
        ipt_b: Tensor,
        N_interpolations: int,
        verbose: bool = True,
    ):
        j_a = self._compute_jacobians(emb_a[0], ipt_a, verbose=verbose)
        j_b = self._compute_jacobians(emb_b[0], ipt_b, verbose=verbose)

        # embeddings of original input and reference
        ex_a = emb_a[0][0].unsqueeze(0).detach()
        ex_b = emb_b[0][0].unsqueeze(0).detach()
        er_a = emb_a[1].unsqueeze(0).detach()
        er_b = emb_b[1].unsqueeze(0).detach()
        # input/intermediate representations of input and reference
        x_a = ipt_a[0].unsqueeze(0).detach()
        x_b = ipt_b[0].unsqueeze(0).detach()
        r_a = ipt_a[-1].unsqueeze(0).detach()
        r_b = ipt_b[-1].unsqueeze(0).detach()

        # deltas for multiplication
        d_a = x_a - r_a
        d_b = x_b - r_b

        # normalizing integrated jacobians
        J_a = (j_a / N_interpolations).to(self.device)
        J_b = (j_b / N_interpolations).to(self.device)

        # reshaping for multiplication
        D_emb, S_a, D_a = J_a.shape
        _, S_b, D_b = J_b.shape
        J_a = J_a.view((D_emb, S_a * D_a)).float()
        J_b = J_b.view((D_emb, S_b * D_b)).float()
        d_a = d_a.view((S_a * D_a)).unsqueeze(1).repeat((1, S_b * D_b))
        d_b = d_b.view((S_b * D_b)).unsqueeze(1).repeat((1, S_b * D_b))

        # multiplication
        J_a, J_b = J_a.cpu(), J_b.cpu()
        d_a, d_b = d_a.cpu(), d_b.cpu()
        ex_a, ex_b = ex_a.cpu(), ex_b.cpu()
        er_a, er_b = er_a.cpu(), er_b.cpu()
        J = torch.mm(J_a.T, J_b)
        A = d_a * J * d_b.T
        # scaling attributions
        scale = self.model.logit_scale.exp()
        # A = A * scale
        norm_a = torch.norm(ex_a)
        norm_b = torch.norm(ex_b)
        A = A / norm_a / norm_b
        A = A.view(S_a, D_a, S_b, D_b)

        # left-hand side terms
        score = self.logit_cos(ex_a.float(), ex_b.float())[0].item()
        ra = self.logit_cos(ex_a.float(), er_b.float())[0].item()
        rb = self.logit_cos(er_a.float(), ex_b.float())[0].item()
        rr = self.logit_cos(er_a.float(), er_b.float())[0].item()
        A_tot = A.sum().item()

        return A, score, ra, rb, rr


class exOpenCLIP(exCLIP):
    def encode_text(self, text: torch.tensor):
        assert len(text.shape) == 2, (
            f"expected text to be a (B, S) tensor, but got {text.shape}"
        )
        text = torch.cat(
            [text, self._make_txt_ref(text_seq_len=text.size(1)).to(self.device)]
        )
        x = self.model.token_embedding(text)  # .type(self.model.dtype)
        # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding  # .type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)  # .type(self.model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if (
            self.attribute
        ):  # expand eot idxs to the number of interpolations in the batch
            txt_eot_idx = text[0].argmax(dim=-1)
            ref_eot_idx = text[1].argmax(dim=-1)
            N = x.shape[0] - 1
            eot_idxs = torch.tensor([txt_eot_idx] * N + [ref_eot_idx])
            x = x[torch.arange(x.shape[0]), eot_idxs]
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.model.text_projection
        if self.shift_emb:
            x = x - x[-1]
        return x[:-1], x[-1]

    def itsm(
        self, text: torch.tensor, image: torch.tensor, clip_txt_padding: bool = True
    ):
        # final sequential text embeddings
        x_txt = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x_txt = x_txt + self.model.positional_embedding
        x_txt = x_txt.permute(1, 0, 2)  # NLD -> LND
        x_txt = self.model.transformer(x_txt)
        x_txt = x_txt.permute(1, 0, 2)  # LND -> NLD
        if clip_txt_padding:
            eot_idx = text.argmax(dim=-1).item()
            x_txt = x_txt[:, : eot_idx + 1, :]
        x_txt = self.model.ln_final(x_txt)

        e_txt = x_txt @ self.model.text_projection
        e_txt = e_txt[0]

        # final image patch embeddings
        x_img = self.model.visual.conv1(image)  # shape = [*, width, grid, grid]
        x_img = x_img.reshape(
            x_img.shape[0], x_img.shape[1], -1
        )  # shape = [*, width, grid ** 2]
        x_img = x_img.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_img = torch.cat(
            [
                self.model.visual.class_embedding.to(x_img.dtype)
                + torch.zeros(
                    x_img.shape[0],
                    1,
                    x_img.shape[-1],
                    dtype=x_img.dtype,
                    device=x_img.device,
                ),
                x_img,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x_img = x_img + self.model.visual.positional_embedding.to(x_img.dtype)
        x_img = self.model.visual.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)  # NLD -> LND
        x_img = self.model.visual.transformer(x_img)
        x_img = x_img.permute(1, 0, 2)  # LND -> NLD
        e_img = x_img @ self.model.visual.proj
        e_img = e_img[0]

        A = e_txt @ e_img.transpose(0, 1)

        # second forward pass to compute score..
        txt_emb, _ = self.encode_text(text)
        img_emb, _ = self.encode_image(image)
        ex_txt = txt_emb[0].unsqueeze(0).detach()
        ex_img = img_emb[0].unsqueeze(0).detach()
        score = self.logit_cos(ex_txt.float(), ex_img.float())[0].item()

        return A, score
