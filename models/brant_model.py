from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.brant_helper.pre_model import TimeEncoder, ChannelEncoder
from models.brant_helper.utils import unwrap_ddp, get_emb
from models.brant_helper.model import MLP
import logging 

log = logging.getLogger(__name__)


@register_model("brant_model")
class BrantModel(BaseModel):
    def __init__(self):
        super(BrantModel, self).__init__()

    def forward(self, inputs, power):

        # Inputs should be of shape (batch_size, num_channels, seq_len, seg_len)
        inputs = inputs.unsqueeze(2) ## (batch_size, num_channels, seg_len) -->  (batch_size, num_channels, 1, seg_len)
        power = power.unsqueeze(2)
        bat_size, ch_num, seq_len, seg_len = inputs.shape

        emb = get_emb(inputs, power, self.encoder_t, self.encoder_ch)
        if self.aggregation_mode in ['mlp_concat', 'linear_concat']:
            emb = emb.reshape(bat_size, ch_num * seq_len * emb.shape[-1])

        logit = self.final_module(emb)

        return logit


    def build_model(self, cfg):
        self.cfg = cfg
        self.aggregation_mode = cfg.aggregation_mode
        self.encoder_t, self.encoder_ch, self.final_module = model_prepare(cfg)


def load_encoder(cfg):
    encoder_t = TimeEncoder(in_dim=cfg.seg_len,
                            d_model=cfg.d_model,
                            dim_feedforward=cfg.dim_feedforward,
                            seq_len=cfg.seq_len,
                            n_layer=cfg.time_ar_layer,
                            nhead=cfg.time_ar_head,
                            band_num=cfg.band_num,
                            project_mode=cfg.input_emb_mode,
                            learnable_mask=cfg.learnable_mask).to(cfg.device)
    encoder_ch = ChannelEncoder(out_dim=cfg.seg_len,
                                d_model=cfg.d_model,
                                dim_feedforward=cfg.dim_feedforward,
                                n_layer=cfg.ch_ar_layer,
                                nhead=cfg.ch_ar_head).to(cfg.device)

    # --------- pretrained model loading ---------
    if cfg.load_pretrained:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.gpu_id}
        t_state_dict = torch.load(cfg.time_encoder_ckpt_path, map_location=map_location)
        ch_state_dict = torch.load(cfg.ch_encoder_ckpt_path, map_location=map_location)
        if cfg.unwrap_ddp:
            t_state_dict = unwrap_ddp(t_state_dict)
            ch_state_dict = unwrap_ddp(ch_state_dict)

        encoder_t.load_state_dict(t_state_dict)
        encoder_ch.load_state_dict(ch_state_dict)
        log.info('----- Pretrained Models Loaded -----\n')

    if cfg.freeze_encoder:
        for param in encoder_t.parameters():
            param.requires_grad = False
        for param in encoder_ch.parameters():
            param.requires_grad = False
    return encoder_t, encoder_ch



def model_prepare(cfg):
    encoder_t, encoder_ch = load_encoder(cfg)

    module = (encoder_t, encoder_ch)
    emb_dim = cfg.d_model

    if cfg.aggregation_mode == 'mlp_concat': 
        final_module = MLP(in_dim=emb_dim * cfg.num_electrodes, out_dim=cfg.output_dim).to(cfg.device)
    elif cfg.aggregation_mode == 'linear_concat':
        final_module = nn.Linear(emb_dim * cfg.num_electrodes, out_features=cfg.output_dim).to(cfg.device)
    elif cfg.aggregation_mode == 'none':
        final_module = MLP(in_dim=emb_dim, out_dim=cfg.output_dim).to(cfg.device)
    module += (final_module, )

    return module
