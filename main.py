import hydra
from omegaconf import DictConfig
import logging
from utils import ins

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    from utils import set_seed
    set_seed(cfg.seed)

    if model_type == 'TrainPolicy':
        from trainer import Policytrainer
        agent = instantiate_class(cfg.agent)
        
        # Train the Policy
        Policytrainer(cfg.train.stock,
                    cfg.train.window_size,
                    cfg.train.episode, 
                    cfg.train.obs_dim,
                    cfg.train.act_dim,
                    cfg.train.agent
                    cfg.train.use_HF,
                    cfg.train.reward_model,
                    cfg.train.device)
        
        # Test the Policy
        Policytester(cfg.train.stock,
                    cfg.train.window_size,
                    cfg.train.episode, 
                    cfg.train.obs_dim,
                    cfg.train.act_dim,
                    cfg.train.agent
                    cfg.train.device)

    elif model_type == 'TrainRewardModel':
        from reward_model import RewardModel
        from util import readTrader

        model = RewardModel(cfg.obs_dim, cfg.act_dim)
        state, reward = readTrader(cfg.rewardmodel. human_feeback, cfg.window_size)

        n = len(reward)

        split = int(0.9*n)

        model = trainRewardModel(model, y_train, x_train
          y_val, x_val, learning_rate=cfg.rewardmodel.lr, epochs=cfg.rewardmodel.epochs, 
          print_out_interval=cfg.rewardmodel.verbose
          stock_name=cfg.rewardmodel.stock_name)
    
    elif model_type = 'UseHumanFeedback':
        from trainer import Policytrainer
        agent = instantiate_class(cfg.agent)
        agent.load(cfg.train.agent_path) # the agent policy trained earlier
        
        # Train the Policy
        Policytrainer(cfg.train.stock,
                    cfg.train.window_size,
                    cfg.train.episode, 
                    cfg.train.obs_dim,
                    cfg.train.act_dim,
                    cfg.train.agent
                    cfg.train.use_HF,
                    cfg.train.reward_model,
                    cfg.train.device)
        
        # Test the Policy
        Policytester(cfg.train.stock,
                    cfg.train.window_size,
                    cfg.train.episode, 
                    cfg.train.obs_dim,
                    cfg.train.act_dim,
                    cfg.train.agent
                    cfg.train.device)


if __name__ == "__main__":
    main()