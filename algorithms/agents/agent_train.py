import torch
from torch.nn.utils import clip_grad_norm_

from utils.utils import time_it
# from utils.batch_crop import batch_crop
# from algorithms.kl_divergence import kl_divergence


class AgentTrain:
    """
    Base class for trainable agents.
    """
    def __init__(
            self,
            model, device,
            learning_rate=3e-4, gamma=0.99, entropy=0.0, clip_grad=0.5,
            normalize_adv=False, returns_estimator='gae',
            gae_lambda=0.9, image_augmentation_alpha=0.0
    ):
        """
        :param model: nn model to train.
        :param device:
        :param lr, gamma, entropy, clip_grad: learning hyper-parameters
        :param normalize_adv: True or False
        :param returns_estimator: '1-step', 'n-step', 'gae'
        :param gae_lambda: gae lambda, optional
        :param image_augmentation_alpha: if > 0 then additional
                                         alpha * D_KL(pi, pi_aug) loss term will be used
        """
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        self.gamma = gamma
        self.entropy = entropy
        self.clip_grad = clip_grad

        self.normalize_adv = normalize_adv
        self.returns_estimator = returns_estimator
        self.gae_lambda = gae_lambda
        self.image_augmentation_alpha = image_augmentation_alpha

    def _one_step_returns(self, next_value, rewards, not_done):
        returns = rewards + self.gamma * not_done * next_value
        return returns

    def _n_step_returns(self, next_value, rewards, not_done):
        rollout_len = rewards.size(0)
        last_value = next_value[-1]
        returns = []
        for t in reversed(range(rollout_len)):
            last_value = rewards[t] + self.gamma * not_done[t] * last_value
            returns.append(last_value)
        returns = torch.stack(returns[::-1])
        return returns

    def _gae(self, value, next_value, rewards, not_done):
        rollout_len = rewards.size(0)
        value = torch.cat([value, next_value[-1:]], dim=0)
        gae = 0
        returns = []
        for t in reversed(range(rollout_len)):
            delta = rewards[t] + self.gamma * not_done[t] * value[t + 1] - value[t]
            gae = delta + self.gamma * self.gae_lambda * not_done[t] * gae
            returns.append(gae + value[t])
        returns = torch.stack(returns[::-1])
        return returns

    def _estimate_returns(self, value, next_value, rewards, is_done):
        not_done = 1.0 - is_done
        with torch.no_grad():  # returns should not have gradients in any case!
            if self.returns_estimator == '1-step':
                returns = self._one_step_returns(next_value, rewards, not_done)
            elif self.returns_estimator == 'n-step':
                returns = self._n_step_returns(next_value, rewards, not_done)
            elif self.returns_estimator == 'gae':
                returns = self._gae(value, next_value, rewards, not_done)
            else:
                raise ValueError('unknown returns estimator')
        return returns.detach()

    @staticmethod
    def _normalize_advantage(advantage):
        mean = advantage.mean()
        std = advantage.std()
        advantage = (advantage - mean) / (std + 1e-8)
        return advantage

    @staticmethod
    def _average_loss(loss, mask):
        if mask is None:
            mask = torch.ones_like(loss)
        return (mask * loss).sum() / mask.sum()

    def _optimize_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        actor_grad_norm = clip_grad_norm_(
            self.model.actor_critic.actor.parameters(), self.clip_grad
        )
        critic_grad_norm = clip_grad_norm_(
            self.model.actor_critic.critic.parameters(), self.clip_grad
        )
        result = {
            'actor_grad_norm': actor_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item()
        }
        if self.model.obs_encoder is not None:
            encoder_gradients = {
                name + '_grad_norm': clip_grad_norm_(child.parameters(), self.clip_grad).item()
                for name, child in self.model.obs_encoder.named_children()
            }
            result.update(encoder_gradients)
        self.optimizer.step()
        return result

    def _compute_returns_advantage(self, values, rewards, is_done):
        value, next_value = values[:-1], values[1:].detach()

        # returns goes into value loss and so must be kept vectorized to train multi-head critic,
        # but advantage used only for policy update and must be summed along last dim.
        returns = self._estimate_returns(value, next_value, rewards, is_done)
        advantage = (returns - value).sum(-1).detach()
        return returns, advantage

    def _train_fn(self, rollout):
        raise NotImplementedError

    # TODO: not supported now
    # def _image_augmentation_loss(self, observation, hidden_state, policy, value):
    #     if self.image_augmentation_alpha > 0.0:
    #         policy, value = policy.detach(), value.detach()
    #
    #         observations_aug = batch_crop(observation)
    #         policy_aug, value_aug, hidden_state_aug = self._get_policy_value_memory(
    #             observations_aug, hidden_state
    #         )
    #         policy_aug, value_aug = policy_aug[:-1], value_aug[:-1]
    #         policy_div = kl_divergence(self.pi_distribution_str, policy, policy_aug).mean()
    #         value_div = 0.5 * ((value - value_aug) ** 2).mean()
    #         augmentation_loss = self.image_augmentation_alpha * (policy_div + value_div)
    #         result_dict = {
    #             'augmentation_policy_div': policy_div.item(),
    #             'augmentation_value_div': value_div.item()
    #         }
    #         return augmentation_loss, result_dict
    #     else:
    #         return 0.0, dict()

    def _update_reward_normalizer_scaler(self, rollout):
        rewards_t = rollout.get('rewards')
        returns_t = rollout.get('returns')
        mask = rollout.mask

        if self.model.reward_normalizer is not None:
            self.model.reward_normalizer.update(rewards_t, mask)
            rewards_t = self.model.reward_normalizer(rewards_t)

        if self.model.reward_scaler is not None:
            self.model.reward_scaler.update(returns_t, mask)
            rewards_t = self.model.reward_scaler(returns_t)
        rollout.set('rewards', rewards_t)

    def _update_obs_emb_normalizer(self, rollout):
        # select all observations except the last to correctly sum with mask.
        observation_t = rollout.get('observations')[:-1]
        mask = rollout.mask

        # update obs normalizer after model.
        if self.model.obs_normalizer is not None:
            self.model.obs_normalizer.update(observation_t, mask)
        if self.model.emb_normalizer is not None:
            observation_t = self.model.obs_normalizer.normalize(observation_t)
            if self.model.recurrent:
                observation_t, _ = self.model.obs_encoder(observation_t, rollout.memory)
            else:
                observation_t = self.model.obs_encoder(observation_t)
            self.model.emb_normalizer.update(observation_t, mask)

    def train_on_rollout(self, rollout, do_train=True):
        rollout.to_tensor(self.model.t, self.device)
        self._update_reward_normalizer_scaler(rollout)

        if do_train:
            train_fn_result, train_fn_time = time_it(self._train_fn)(rollout)
            if isinstance(train_fn_result, tuple):
                result_log, time_log = train_fn_result
            else:  # i.e. result is one dict
                result_log = train_fn_result
                time_log = dict()
            time_log['train_on_rollout'] = train_fn_time
        else:
            result_log, time_log = {}, {}

        self._update_obs_emb_normalizer(rollout)

        return result_log, time_log
