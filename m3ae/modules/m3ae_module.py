import gc
import math
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import RobertaConfig, RobertaModel, RobertaForCausalLM, AutoTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

from m3ae.modules import objectives, m3ae_utils
from m3ae.modules import prediction_heads
from m3ae.modules.language_encoders.bert_model import BertCrossLayer, GAT_module
from m3ae.modules.m3ae_utils import init_weights
from m3ae.modules.vision_encoders import swin_transformer as swin
from m3ae.modules.vision_encoders.clip_model import build_model, adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

from transformers import RobertaTokenizerFast, DataCollatorForWholeWordMask, BatchEncoding
from m3ae.modules.mambapy.mamba import Mamba, MambaConfig, AdaptiveKGFusion


class M3AETransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.is_clip = ('swin' not in config['vit'])

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        elif 'bert' in config['tokenizer']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            raise ValueError

        resolution_after = config['image_size']
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()

        if self.is_clip:
            self.vision_encoder = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vision_encoder = getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
            self.vision_pooler = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config['tokenizer']:
            self.language_encoder = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.language_encoder = BertModel.from_pretrained(config['tokenizer'])

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.language_proj = nn.Linear(config['input_text_embed_size'], config['embed_dim'])
        self.multi_modal_language_proj.apply(init_weights)

        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.vision_proj = nn.Linear(config['input_text_embed_size'], config['embed_dim'])
        self.multi_modal_vision_proj.apply(init_weights)

        for p in self.multi_modal_language_proj.parameters():
            p.requires_grad = False
        for p in self.language_proj.parameters():
            p.requires_grad = False
        for p in self.multi_modal_vision_proj.parameters():
            p.requires_grad = False
        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)

        used_device = f"cuda:{config['gpu_ids']}" if torch.cuda.is_available() else "cpu"
        self.latent_prompt_size = config['latent_prompt_size']
        self.organ_disease_feat_size = 577
        self.GAT_feat = 32
        self.adj_feat = torch.load(config["adj_feat_path"], map_location=used_device)
        self.organ_disease_feat = torch.load(config["organ_disease_feat"], map_location=used_device)

        self.GAT_layer = GAT_module(config["hidden_size"])
        self.organ_average = nn.Conv1d(self.organ_disease_feat_size, self.GAT_feat, 1)
        self.organ_average.apply(init_weights)

        self.qformer = Blip2Qformer()

        mamba_config = MambaConfig(d_model=config["hidden_size"], n_layers=2)
        self.multi_modal_vision_mamba = Mamba(mamba_config)
        self.multi_modal_graph_mamba = Mamba(mamba_config)
        self.multi_modal_kg_fusion = AdaptiveKGFusion(config["hidden_size"])

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)

        itc = False
        if itc:
            self.temp = nn.Parameter(torch.ones(1) * config['temp'])
            self.queue_size = config['queue_size']
            self.momentum = config['momentum']

            self.vision_encoder_m = build_model(config['vit'], resolution_after=resolution_after)
            self.vision_proj_m = nn.Linear(config['input_image_embed_size'], config['embed_dim'])

            self.language_encoder_m = RobertaModel.from_pretrained(config['tokenizer'])
            self.language_proj_m = nn.Linear(config['input_text_embed_size'], config['embed_dim'])

            self.model_pairs = [
                [self.vision_encoder, self.vision_encoder_m],
                [self.vision_proj, self.vision_proj_m],
                [self.language_encoder, self.language_encoder_m],
                [self.language_proj, self.language_proj_m],
            ]

            self.copy_params()

            self.register_buffer("image_queue", torch.randn(config['embed_dim'], self.queue_size))
            self.register_buffer("text_queue", torch.randn(config['embed_dim'], self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_head = prediction_heads.MLMHead(bert_config)
            self.mlm_head.apply(init_weights)
        if config["loss_names"]["mim"] > 0:
            self.mim_head = prediction_heads.MIMHead(config)
            self.mim_head.apply(init_weights)
        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["irtr"] > 0:
            self.itm_head = prediction_heads.ITMHead(config["hidden_size"] * 2)
            self.itm_head.apply(init_weights)

        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(
                    state_dict,
                    after=resolution_after,
                    patch_size=self.hparams.config['patch_size']
                )
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)
            del ckpt, state_dict
            gc.collect()

        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqa_label_size"]
            self.vqa_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_head.apply(init_weights)

        if self.hparams.config["loss_names"]["cls"] > 0:
            ms = self.hparams.config["melinda_label_size"][self.hparams.config["label_column_name"]]
            self.cls_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.cls_head.apply(init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.irtr_head = nn.Linear(hs * 2, 1)
            self.irtr_head.weight.data = self.itm_head.fc.weight.data[1:, :]
            self.irtr_head.bias.data = self.itm_head.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_head.parameters():
                p.requires_grad = False

        m3ae_utils.set_metrics(self)
        self.current_tasks = list()

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(
                state_dict,
                after=resolution_after,
                patch_size=self.hparams.config['patch_size']
            )
            self.load_state_dict(state_dict, strict=False)
            del ckpt, state_dict
            gc.collect()

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        pos_embed = self.vision_encoder.visual.positional_embedding.unsqueeze(0).to(x)

        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x += pos_embed[:, 1:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        p = self.hparams.config["patch_size"]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=True,
            itc=False,
            unimodal=False,
    ):
        ret = dict()

        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key][0]
        device = img.device
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"].to(device)
        text_labels = batch[f"text_labels{do_mlm}"].to(device)
        text_masks = batch[f"text_masks"].to(device)

        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
        text_input_shape = text_masks.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_input_shape, device)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]

        if mask_image:
            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)
            uni_modal_image_feats, mim_masks, mim_ids_restore = self.random_masking(
                uni_modal_image_feats, self.hparams.config["mim_prob"]
            )
            uni_modal_image_feats = self.vision_encoder.forward_trans(uni_modal_image_feats)
            ret["mim_masks"] = mim_masks
            ret["mim_ids_restore"] = mim_ids_restore
        else:
            uni_modal_image_feats = self.vision_encoder(img)

        image_masks = torch.ones(
            (uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)),
            dtype=torch.long,
            device=device
        )
        extended_image_masks = self.language_encoder.get_extended_attention_mask(
            image_masks, image_masks.size(), device
        )

        if itc:
            text_feat = F.normalize(self.language_proj(uni_modal_text_feats[:, 0, :]), dim=-1)
            image_feat = F.normalize(self.vision_proj(uni_modal_image_feats[:, 0, :]), dim=-1)
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)

            alpha = 0.4

            with torch.no_grad():
                self._momentum_update()

                image_embeds_m = self.vision_encoder_m(img)
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

                text_output_m = self.language_encoder_m.embeddings(input_ids=text_ids)
                text_input_shape = text_masks.size()
                extended_text_masks = self.language_encoder.get_extended_attention_mask(
                    text_masks, text_input_shape, device
                )
                for layer in self.language_encoder.encoder.layer:
                    text_output_m = layer(text_output_m, extended_text_masks)[0]
                text_output_m = self.multi_modal_language_proj(text_output_m)
                text_feat_m = F.normalize(self.language_proj_m(text_output_m[:, 0, :]), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(img.device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

            loss_itc = (loss_i2t + loss_t2i) / 2

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        else:
            loss_itc = torch.tensor(0.0, device=device)

        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        ret["attentions"] = {"text2image_attns": [], "image2text_attns": []} if output_attentions else None
        x, y = uni_modal_text_feats, uni_modal_image_feats

        if self.adj_feat.device != device:
            self.adj_feat = self.adj_feat.to(device)
        if self.organ_disease_feat.device != device:
            self.organ_disease_feat = self.organ_disease_feat.to(device)

        adj_feat = self.adj_feat
        organ_disease_feat = self.organ_disease_feat.long()

        vocab_size = self.language_encoder.embeddings.word_embeddings.num_embeddings
        organ_disease_feat = torch.clamp(organ_disease_feat, 0, vocab_size - 1)

        if organ_disease_feat.dim() > 1:
            organ_disease_feat = organ_disease_feat.view(-1)
        if organ_disease_feat.size(0) != 577:
            organ_disease_feat = organ_disease_feat[:577]

        organ_disease_feat = self.language_encoder.embeddings.word_embeddings(organ_disease_feat)
        if organ_disease_feat.dim() == 2:
            organ_disease_feat = organ_disease_feat.unsqueeze(0).expand(x.size(0), -1, -1)

        GAT_feat = self.GAT_layer(x, organ_disease_feat, adj_feat)
        GAT_feat = self.organ_average(GAT_feat)

        y, loss_itc = self.qformer(x, y)
        if torch.isnan(loss_itc) or torch.isinf(loss_itc):
            loss_itc = torch.tensor(0.0, device=device)

        x1, y = self.multi_modal_vision_mamba(x, y, mask_v=text_masks)
        x2, k = self.multi_modal_graph_mamba(x, GAT_feat, mask_v=text_masks)

        multi_modal_text_feat_i, multi_modal_text_feat_k, multi_modal_image_feats, knowledge_feat = x1, x2, y, k

        multi_modal_text_feats, multi_modal_image_feats = self.multi_modal_kg_fusion(
            multi_modal_text_feat_i,
            multi_modal_text_feat_k,
            knowledge_feat,
            z_t2z=multi_modal_image_feats,
        )
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(multi_modal_text_feats)

        if self.is_clip:
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(multi_modal_image_feats)
        else:
            avg_image_feats = self.vision_pooler(multi_modal_image_feats.transpose(1, 2)).view(
                multi_modal_image_feats.size(0), 1, -1
            )
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(avg_image_feats)

        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
        ag_loss = torch.tensor(0.0, device=device)

        ret.update({
            "images": img,
            "patched_images": self.patchify(img) if mask_image else None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_image_feats": multi_modal_image_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
            "itc_loss": loss_itc,
            "ag_loss": ag_loss
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)
        total_loss = sum([
            v * self.hparams.config["loss_names"][k.replace("_loss", "")]
            for k, v in output.items() if "loss" in k
        ])
        return total_loss

    def on_after_backward(self):
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                p.grad.clamp_(-1.0, 1.0)

    def on_train_epoch_end(self, outs=None):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self, outs=None):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def on_test_epoch_end(self, outs=None):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        if is_dist_avail_and_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat

        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        weight_sigmoid = torch.Tensor(size_out * 2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))
        nn.init.normal_(self.weight_sigmoid)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2