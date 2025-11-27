# uct_mcts.py
from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame
from other_algo.heuristic import go_heuristic_evaluation

import numpy as np


class UCTMCTSConfig(MCTSConfig):
    def __init__ (
            self,
            n_rollout: int = 1,
            *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout


class UCTMCTS:
    def __init__ (self, init_env: BaseGame, config: UCTMCTSConfig, root: MCTSNode = None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()

    def init_tree (self, init_env: BaseGame):
        env = init_env.fork()
        self.root = MCTSNode(
            action = None, env = env, reward = 0,
        )

    def get_subtree (self, action: int):
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None

    def uct_action_select (self, node: MCTSNode) -> int:
        total_visits = np.sum(node.child_N_visit)
        uct_values = np.zeros(node.n_action, dtype = np.float32)

        # 只考虑有效动作
        valid_actions = np.where(node.env.action_mask == 1)[0]
        if len(valid_actions) == 0:
            return -1  # 如果没有有效动作，返回 -1

        for action in valid_actions:
            if node.child_N_visit[action] == 0:
                uct_values[action] = INF
            else:
                exploitation = node.child_V_total[action] / node.child_N_visit[action]
                exploration = self.config.C * np.sqrt(np.log(total_visits) / node.child_N_visit[action])
                uct_values[action] = exploitation + exploration

        return np.argmax(uct_values)

    def backup (self, node: MCTSNode, value: float) -> None:
        current_node = node
        parent_node = current_node.parent
        while parent_node is not None:
            parent_node.child_N_visit[current_node.action] += 1
            parent_node.child_V_total[current_node.action] += value
            value = -value
            current_node = parent_node
            parent_node = parent_node.parent

    def rollout (self, node: MCTSNode) -> float:
        env = node.env.fork()
        while not env.ended:
            # 获取有效动作掩码
            valid_actions = np.where(env.action_mask == 1)[0]
            if len(valid_actions) == 0:
                break  # 如果没有有效动作，直接退出
            # 从有效动作中随机选择一个动作
            action = np.random.choice(valid_actions)
            _, reward, _ = env.step(action)
        return reward

    def pick_leaf (self) -> MCTSNode:
        node = self.root
        while not node.done:
            # 获取有效动作掩码
            valid_actions = np.where(node.env.action_mask == 1)[0]
            if len(valid_actions) == 0:
                break  # 如果没有有效动作，直接退出
            # 使用 UCT 选择有效动作
            action = self.uct_action_select(node)
            if action == -1:
                break  # 如果没有有效动作，直接退出
            # 如果选择的动作无效，则从有效动作中随机选择一个
            if not node.has_child(action) and node.env.action_mask[action] == 0:
                action = np.random.choice(valid_actions)
            if not node.has_child(action):
                return node.add_child(action)
            node = node.get_child(action)
        return node

    def get_policy (self, node: MCTSNode = None) -> np.ndarray:
        if node is None:
            node = self.root
        # 只考虑有效动作
        valid_actions = np.where(node.env.action_mask == 1)[0]
        if len(valid_actions) == 0:
            return np.zeros(node.n_action)  # 如果没有有效动作，返回全零策略
        # 计算有效动作的访问次数
        valid_visits = node.child_N_visit[valid_actions]
        total_visits = np.sum(valid_visits)
        if total_visits == 0:
            # 如果没有访问记录，则均匀分布
            policy = np.zeros(node.n_action)
            policy[valid_actions] = 1.0 / len(valid_actions)
            return policy
        # 归一化有效动作的访问次数
        policy = np.zeros(node.n_action)
        policy[valid_actions] = valid_visits / total_visits
        return policy

    def search (self):
        step_count = 0
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            if leaf.done:
                value = leaf.reward
            else:
                # value = 0
                # for _ in range(self.config.n_rollout):
                #     value += self.rollout(leaf)
                # value /= self.config.n_rollout
                if step_count < 10:
                    # 前十步使用 rollout
                    value = sum(self.rollout(leaf) for _ in range(self.config.n_rollout)) / self.config.n_rollout
                else:
                    # 十步后使用 heuristic_evaluate
                    value = self.heuristic_evaluate(leaf)
            self.backup(leaf, value)
            step_count += 1  # 计步
        return self.get_policy(self.root)

    def heuristic_evaluate(self, node: MCTSNode) -> float:
        return go_heuristic_evaluation(node.env)