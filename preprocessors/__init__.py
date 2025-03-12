from .stft import STFTPreprocessor
from .morelet_preprocessor import MoreletPreprocessor
from .superlet_preprocessor import SuperletPreprocessor
from .wav_preprocessor import WavPreprocessor
from .downsample_wav_preprocessor import DownsampleWavPreprocessor
from .superlet import superlet
from .spec_pretrained import SpecPretrained
from .multi_elec_spec_pretrained import MultiElecSpecPretrained
from .spec_pooled import SpecPooled
from .chronos import Chronos
from .chronos_scaling_preprocessor import ChronosScalingPreprocessor
from .identity_preprocessor import IdentityPreprocessor

__all__ = ["STFTPreprocessor",
           "MoreletPreprocessor",
           "SuperletPreprocessor",
           "superlet",
           "WavPreprocessor",
           "DownsampleWavPreprocessor",
           "SpecPretrained",
           "MultiElecSpecPretrained",
           "SpecPooled",
           "Chronos",
           "ChronosScalingPreprocessor"
           "IdentityPreprocessor"
          ]

def build_preprocessor(preprocessor_cfg):
    if preprocessor_cfg.name == "stft":
        extracter = STFTPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "superlet":
        extracter = SuperletPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "wav_preprocessor":
        extracter = WavPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "downsample_wav_preprocessor":
        extracter = DownsampleWavPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "spec_pretrained":
        extracter = SpecPretrained(preprocessor_cfg)
    elif preprocessor_cfg.name == "multi_elec_spec_pretrained":
        extracter = MultiElecSpecPretrained(preprocessor_cfg)
    elif preprocessor_cfg.name == "spec_pooled_preprocessor":
        extracter = SpecPooled(preprocessor_cfg)
    elif preprocessor_cfg.name == "chronos":
        extracter = Chronos(preprocessor_cfg)
    elif preprocessor_cfg.name == "chronos_scaling_preprocessor":
        extracter = ChronosScalingPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "identity_preprocessor":
        extracter = IdentityPreprocessor(preprocessor_cfg)
    else:
        raise ValueError("Specify preprocessor")
    return extracter
