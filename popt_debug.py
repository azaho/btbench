from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.info("Load PopT")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    #model = TheModelClass(*args, **kwargs)
    #model.load_state_dict(torch.load(PATH, weights_only=True))
    #model.eval()
