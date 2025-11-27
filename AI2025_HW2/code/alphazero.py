from env import *
from env.base_env import *
from torch.nn import Module
from model.linear_model import NumpyLinearModel
from model.linear_model_trainer import NumpyLinearModelTrainer, NumpyModelTrainingConfig
from mcts import puct_mcts

import numpy as np
import random
import torch
import copy
from tqdm import tqdm
from random import shuffle
from players import PUCTPlayer, RandomPlayer, AlphaBetaPlayer

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)


class AlphaZeroConfig():
    def __init__(
        self, 
        n_train_iter:int=300,
        n_match_train:int=20,
        n_match_update:int=20,
        n_match_eval:int=20,
        max_queue_length:int=8000,
        update_threshold:float=0.501,
        n_search:int=200, 
        temperature:float=1.0, 
        C:float=1.0,
        checkpoint_path:str="checkpoint"
    ):
        self.n_train_iter = n_train_iter
        self.n_match_train = n_match_train
        self.max_queue_length = max_queue_length
        self.n_match_update = n_match_update
        self.n_match_eval = n_match_eval
        self.update_threshold = update_threshold
        self.n_search = n_search
        self.temperature = temperature
        self.C = C
        
        self.checkpoint_path = checkpoint_path

class AlphaZero:
    def __init__(self, env:BaseGame, net:NumpyLinearModelTrainer, config:AlphaZeroConfig):
        self.env = env
        self.net = net
        self.last_net = net.copy()
        self.config = config
        self.mcts_config = puct_mcts.MCTSConfig(
            C=config.C, 
            n_search=config.n_search, 
            temperature=config.temperature
        )
        self.mcts_config.with_noise = False
        self.mcts = None
        self.train_eamples_queue = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
    
    def execute_episode(self):
        train_examples = []
        env = self.env.fork()
        state = env.reset()
        config = copy.copy(self.mcts_config)
        config.with_noise = True
        mcts = puct_mcts.PUCTMCTS(env, self.net, config)
        while True:
            player = env.current_player
            # MCTS self-play
            ########################
            # TODO: your code here #
            policy = mcts.search() # compute policy with mcts

            symmetries = get_symmetries(state, policy) # rotate&flip the data&policy
            train_examples += [(x[0], x[1], player) for x in symmetries]

            # 选择动作：只从 valid_actions 中按概率选
            action = np.random.choice(len(policy), p = policy)
            next_state, reward, done = env.step(action) # apply the action to env
            if done:
                train_examples = [
                    (env.compute_canonical_form_obs(x[0], x[2]), x[1], reward * (-1 if x[2] != player else 1))
                    for x in train_examples
                ]
                return train_examples # record all data
                # tips: use env.compute_canonical_form_obs to transform the observation into BLACK's perspective

            mcts = mcts.get_subtree(action) # update mcts (you can use get_subtree())
            state = next_state
            ########################
    
    def evaluate(self):
        player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
        # baseline_player = AlphaBetaPlayer()
        baseline_player = RandomPlayer()
        result = multi_match(self.env, player, baseline_player, self.config.n_match_eval)
        logger.info(f"[EVALUATION RESULT]: win{result[0][0]}, lose{result[0][1]}, draw{result[0][2]}")
        logger.info(f"[EVALUATION RESULT]:(first)  win{result[1][0]}, lose{result[1][1]}, draw{result[1][2]}")
        logger.info(f"[EVALUATION RESULT]:(second) win{result[2][0]}, lose{result[2][1]}, draw{result[2][2]}")
    
    def learn(self):
        for iter in range(1, self.config.n_train_iter + 1):
            logger.info(f"------ Start Self-Play Iteration {iter} ------")
            
            # collect new examples
            T = tqdm(range(self.config.n_match_train), desc="Self Play")
            cnt = ResultCounter()
            for _ in T:
                episode = self.execute_episode()
                self.train_eamples_queue += episode
                cnt.add(episode[0][-1], 1)
            logger.info(f"[NEW TRAIN DATA COLLECTED]: {str(cnt)}")
            
            # pop old examples
            if len(self.train_eamples_queue) > self.config.max_queue_length:
                self.train_eamples_queue = self.train_eamples_queue[-self.config.max_queue_length:]
            
            # shuffle examples for training
            train_data = copy.copy(self.train_eamples_queue)
            shuffle(train_data)
            logger.info(f"[TRAIN DATA SIZE]: {len(train_data)}")
            
            # save current net to last_net
            self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
            self.last_net.load_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
            
            # train current net
            self.net.train(train_data)
            
            # evaluate current net
            env = self.env.fork()
            env.reset()
            
            last_mcts_player = PUCTPlayer(self.mcts_config, self.last_net, deterministic=True)
            current_mcts_player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
            
            result = multi_match(self.env, last_mcts_player, current_mcts_player, self.config.n_match_update)[0]
            # win_rate = result[1] / sum(result)
            total_win_lose = result[0] + result[1]
            win_rate = result[1] / total_win_lose if total_win_lose > 0 else 1
            logger.info(f"[EVALUATION RESULT]: currrent_win{result[1]}, last_win{result[0]}, draw{result[2]}; win_rate={win_rate:.3f}")
            
            if win_rate > self.config.update_threshold:
                self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
                logger.info(f"[ACCEPT NEW MODEL]")
                self.evaluate()
            else:
                self.net.load_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')
                logger.info(f"[REJECT NEW MODEL]")


if __name__ == "__main__":
    from env import *
    import torch
    
    MASTER_SEED = 0
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)
    
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Linear
    config = AlphaZeroConfig(
        n_train_iter=50,
        n_match_train=20,
        n_match_update=20,
        n_match_eval=20,
        max_queue_length=160000,
        update_threshold=0.500,
        n_search=240, 
        temperature=1.0, 
        C=1.0,
        checkpoint_path="checkpoint/linear_7x7_exfeat_norm_p1"
    )
    model_training_config = NumpyModelTrainingConfig(
        epochs=10,
        batch_size=128,
        lr=0.0001,
        weight_decay=0.001
    )
    assert config.n_match_update % 2 == 0
    assert config.n_match_eval % 2 == 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Go Game with extended observation
    env = GoGame(7, obs_mode="extra_feature")
    
    # oringin observation, only the n*n game board
    # Use this for debug, if needed
    # env = GoGame(7) 
    
    def base_function(X: np.ndarray) -> np.ndarray:
        return X
    
    net = NumpyLinearModel(env.observation_size, env.action_space_size, None, device=device)
    net = NumpyLinearModelTrainer(env.observation_size, env.action_space_size, net, model_training_config)
    
    alphazero = AlphaZero(env, net, config)
    alphazero.learn()