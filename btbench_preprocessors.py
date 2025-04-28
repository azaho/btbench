from torch import nn

def build_preprocessor(name):
    if name=="identity":
        return IdentityPreprocessor()
    elif name=="multi_elec_spec_pretrained":
        cfg = {"spec_name": stft,
               "freq_channel_cutoff": 40,
               "nperseg": 400,
               "noverlap": 350,
               "normalizing": zscore,
               "upstream_ckpt": "/storage/czw/self_supervised_seeg/pretrained_weights/stft_large_pretrained.pth" } #TODO hardcode
        import pdb; pdb.set_trace()
    else:
        raise ValueError("Preprocessor does not exist")

class IdentityPreprocessor(nn.Module):
    def __init__(self):
        super(IdentityPreprocessor, self).__init__()

    def forward(self, x):
        return x

class IdentityPreprocessor(nn.Module):
    def __init__(self):
        super(IdentityPreprocessor, self).__init__()

    def forward(self, x):
        return x
