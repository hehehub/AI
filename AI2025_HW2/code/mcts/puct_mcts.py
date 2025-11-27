from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__ (self, init_env: BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root: MCTSNode = None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()

    def init_tree (self, init_env: BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action = None, env = env, reward = 0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)

    def get_subtree (self, action: int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None

    def puct_action_select(self, node:MCTSNode):
       # select the best action based on PUCB when expanding the tree

        ########################
        # TODO: your code here #
        ########################
       # 根据PUCB选择最优的动作
        if np.sum(node.child_N_visit) == 0:
            return np.random.choice(np.where(node.action_mask == 1)[0])

        weights = (
                node.child_V_total / (node.child_N_visit + 1e-8)  # Avoid division by zero
                + self.config.C * node.child_priors * np.sqrt(np.sum(node.child_N_visit)) / (node.child_N_visit + 1)
        )
        # Mask out invalid actions
        weights[node.action_mask == 0] = -INF
        return np.argmax(weights)
        ########################

    def backup(self, node:MCTSNode, value):
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path

        ########################
        # TODO: your code here #
        ########################
        current_node = node
        parent_node = current_node.parent
        while parent_node is not None:
            parent_node.child_N_visit[current_node.action] += 1
            parent_node.child_V_total[current_node.action] += value
            value = -value  # 因为是对抗游戏，所以值取反
            current_node = parent_node
            parent_node = parent_node.parent
        ########################

    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended

        ########################
        # TODO: your code here #
        ########################
        node = self.root
        while not node.done:
            # 获取有效动作掩码
            valid_actions = np.where(node.env.action_mask == 1)[0]
            if len(valid_actions) == 0:
                break  # 如果没有有效动作，直接退出
            # 使用 PUCT 选择有效动作
            action = self.puct_action_select(node)
            if action == -1:
                break  # 如果没有有效动作，直接退出
            # 如果选择的动作无效，则从有效动作中随机选择一个
            if not node.has_child(action) and node.env.action_mask[action] == 0:
                action = np.random.choice(valid_actions)
            if not node.has_child(action):
                return node.add_child(action)
            node = node.get_child(action)
        return node
        ########################

    def get_policy(self, node:MCTSNode = None):
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action

        ########################
        # TODO: your code here #
        ########################
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
        ########################

    def search(self):
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                ########################
                # TODO: your code here #
                ########################
                value = leaf.reward
                ########################
            else:
                ########################
                # TODO: your code here #
                ########################
                # NOTE: you should compute the policy and value
                #       using the value&policy model!
                obs = leaf.env.observation
                policy, value = self.model.predict(leaf.env.compute_canonical_form_obs(obs, leaf.env.current_player))
                leaf.set_prior(policy)
                if leaf.env.current_player != self.root.env.current_player:
                    value = -value  # 因为对抗游戏，所以值取反
                ########################
            self.backup(leaf, value)

        return self.get_policy(self.root)