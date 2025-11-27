import numpy as np
from env.base_env import *
from other_algo.heuristic import go_heuristic_evaluation

MAX_VALUE = 1 

class AlphaBetaSearchWithHeuristic:

    def __init__(self, depth=2):
        self.cnt = 0
        self.depth_limit = depth
        
    def max_value(self, game_state:BaseGame, alpha, beta, depth):
        self.cnt += 1 # for debug
        
        ########################
        # 如果达到深度限制或游戏结束，返回启发式评估值
        if depth == 0 or game_state.ended:
            return go_heuristic_evaluation(game_state)

        v = -np.inf
        valid_moves_mask = game_state.action_mask
        valid_moves = np.where(valid_moves_mask == 1)[0]

        for a in valid_moves:
            # 复制游戏状态并执行动作
            next_state = game_state.fork()
            _, r, _ = next_state.step(a)

            # 如果游戏结束，直接返回奖励值
            if r != NOTEND:
                v = max(v, r)
            else:
                # 如果游戏未结束，递归搜索，并使用 -max_value 作为对手的值
                # 同时交换 alpha 和 beta，因为玩家已经改变
                v = max(v, -self.max_value(next_state, -beta, -alpha, depth - 1))

            # 剪枝
            # 如果 -v <= beta，则通过 beta 剪枝（对手的最大值）
            # 如果 v == MAX_VALUE，v 不能再大
            if -v <= beta or v == MAX_VALUE:
                return v

            # 更新 alpha，即当前玩家的最大值
            alpha = max(alpha, v)

        return v
        ########################

    def get_best_move(self, game_state:BaseGame):
        best_move, best_value = -1, -np.inf
        alpha, beta = -np.inf, -np.inf
        valid_moves_mask = game_state.action_mask
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            next_state = game_state.fork()
            board, r, _ = next_state.step(a)
            v = r if r !=NOTEND else -self.max_value(next_state, beta, alpha, self.depth_limit)
            if v > best_value:
                best_value = v
                best_move = a
        # print(self.cnt) # use this to check the number of nodes expanded
        self.cnt = 0
        return best_move


class AlphaBetaSearch:

    def __init__(self):
        self.cnt = 0

    def max_value(self, game_state:BaseGame, alpha, beta):
        self.cnt += 1 # for debug
        v = -np.inf
        valid_moves_mask = game_state.action_mask
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            # copy the game state and take the action
            next_state = game_state.fork()
            _, r, _ = next_state.step(a, return_obs=False)
            
            # check result
            if r != NOTEND: # if game is end
                v = max(v, r)
            else: 
                # if game is not end, recursive search, and use -max_value of next state
                # also swap alpha and beta as the player has changed
                v = max(v, - self.max_value(next_state, beta, alpha))
                
            # pruning; 
            # if -v <= beta, cut by beta (the maximum value of the opponent)
            # if v == MAX_VALUE, v can not be larger
            if -v <= beta or v == MAX_VALUE: 
                return v
            
            # update alpha, which is the maximum value of current player
            alpha = max(alpha, v)
        return v

    def get_best_move(self, game_state:BaseGame):
        best_move, best_value = -1, -np.inf
        alpha, beta = -np.inf, -np.inf
        valid_moves_mask = game_state.action_mask
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            next_state = game_state.fork()
            _, r, _ = next_state.step(a, return_obs=False)
            v = r if r !=NOTEND else -self.max_value(next_state, beta, alpha)
            if v > best_value:
                best_value = v
                best_move = a
        # print(self.cnt) # use this to check the number of nodes expanded
        self.cnt = 0
        return best_move
