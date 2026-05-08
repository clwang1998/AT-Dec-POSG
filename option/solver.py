from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from .config import (
    OptionExecutionConfig,
    OptionMarketScenario,
    OptionTrainingConfig,
    default_scenarios,
)
from .env import MultiOptionExecutionEnv
from .models import ATDecOptionModel


@dataclass
class EpisodeRollout:
    obs: np.ndarray
    clean_obs: np.ndarray
    next_obs: np.ndarray
    privileged: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    belief_targets: np.ndarray
    sender_targets: np.ndarray
    receiver_targets: np.ndarray
    module_c_targets: np.ndarray
    agent_indices: np.ndarray


def _opponent_value_proxy(info: Dict[str, object]) -> float:
    # Proxy in the opponent (market-side) coordinates: larger is worse for the
    # execution team and should therefore induce a non-positive shaping term.
    return float(
        float(info["team_shortfall"])
        + float(info["inventory_penalty"])
        + float(info["risk_penalty"])
        + float(info["final_penalty"])
    )


def _compute_shifted_minimax_bonus(
    training_config: OptionTrainingConfig,
    info: Dict[str, object],
    done: bool,
    training: bool,
) -> float:
    if not training or not training_config.use_module_c:
        return 0.0
    if info.get("scenario_role") != "exploiter":
        return 0.0
    alpha = float(training_config.minimax_alpha)
    if alpha <= 0.0:
        return 0.0
    gamma = float(np.clip(training_config.minimax_gamma, 0.0, 1.0))
    if gamma <= 0.0:
        return 0.0
    done_factor = 0.0 if done else 1.0
    if done_factor <= 0.0:
        return 0.0
    value_floor = float(training_config.minimax_value_floor)
    shifted_value = max(_opponent_value_proxy(info) - value_floor, 0.0)
    return -alpha * gamma * done_factor * shifted_value


def _compute_module_c_aux_target(
    training_config: OptionTrainingConfig,
    info: Dict[str, object],
) -> float:
    if not training_config.use_module_c:
        return 0.0

    role = str(info.get("scenario_role", "main"))
    role_scale = {
        "main": 0.0,
        "snapshot": 0.5,
        "exploiter": 1.0,
    }.get(role, 0.0)
    if role_scale <= 0.0:
        return 0.0

    shifted_value = max(
        _opponent_value_proxy(info) - float(training_config.minimax_value_floor),
        0.0,
    )
    proxy_target = shifted_value / (1.0 + shifted_value)
    coordination_stress = float(np.clip(info.get("coordination_stress", 0.0), 0.0, 1.0))
    return float(np.clip(role_scale * 0.5 * (proxy_target + coordination_stress), 0.0, 1.0))


class ScenarioLeague:
    """Lightweight league over adversarial market regimes for Module C."""

    def __init__(
        self,
        training: OptionTrainingConfig,
        seed: int,
        scenarios: Optional[List[OptionMarketScenario]] = None,
    ) -> None:
        self.training = training
        self.rng = np.random.default_rng(seed)
        all_scenarios = scenarios or default_scenarios()
        self.main = [s for s in all_scenarios if s.role == "main"] or [OptionMarketScenario()]
        self.snapshots = [s for s in all_scenarios if s.role == "snapshot"]
        self.exploiters = [s for s in all_scenarios if s.role == "exploiter"]

    def sample(self) -> OptionMarketScenario:
        main_p, snapshot_p, exploiter_p = self.training.role_probs()
        available = []
        weights = []
        if self.main:
            available.extend(self.main)
            weights.extend([main_p / len(self.main)] * len(self.main))
        if self.snapshots:
            available.extend(self.snapshots)
            weights.extend([snapshot_p / len(self.snapshots)] * len(self.snapshots))
        if self.exploiters:
            available.extend(self.exploiters)
            weights.extend([exploiter_p / len(self.exploiters)] * len(self.exploiters))
        probs = np.array(weights, dtype=np.float64)
        probs /= probs.sum()
        return available[int(self.rng.choice(np.arange(len(available)), p=probs))]

    def record(self, episode_idx: int, metrics: Dict[str, float]) -> None:
        if episode_idx == 0 or episode_idx % 50 != 0:
            return
        hardness = np.clip(
            metrics["implementation_shortfall"] / max(abs(metrics["reward"]), 1.0),
            0.0,
            1.0,
        )
        if self.exploiters:
            base = self.exploiters[-1]
            self.exploiters[-1] = OptionMarketScenario(
                name=f"exploiter_refresh_{episode_idx}",
                role="exploiter",
                spread_multiplier=min(1.6, base.spread_multiplier + 0.05 * (1.0 - hardness)),
                impact_multiplier=min(1.6, base.impact_multiplier + 0.05 * (1.0 - hardness)),
                volatility_multiplier=min(
                    1.5, base.volatility_multiplier + 0.03 * (1.0 - hardness)
                ),
                pressure_bias=min(0.30, base.pressure_bias + 0.02 * (1.0 - hardness)),
                liquidity_decay=min(0.20, base.liquidity_decay + 0.01 * (1.0 - hardness)),
            )


class OptionPPOSolver:
    def __init__(
        self,
        env_config: Optional[OptionExecutionConfig] = None,
        training_config: Optional[OptionTrainingConfig] = None,
        device: Optional[str] = None,
        seed: int = 7,
    ) -> None:
        self.env_config = env_config or OptionExecutionConfig(seed=seed)
        self.training = training_config or OptionTrainingConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = ATDecOptionModel(
            obs_dim=self.env_config.obs_dim,
            privileged_dim=self.env_config.privileged_dim,
            action_dim=self.env_config.action_dim,
            belief_dim=self.env_config.belief_dim,
            hidden_dim=self.training.hidden_dim,
            coordination_dim=self.training.module_b_coord_dim,
        ).to(self.device)
        self.multitask_log_vars: Optional[torch.nn.ParameterDict] = None
        task_names = self._multitask_task_names()
        if self.training.use_multitask_loss_balancer and len(task_names) > 1:
            init = float(self.training.multitask_log_var_init)
            self.multitask_log_vars = torch.nn.ParameterDict(
                {
                    name: torch.nn.Parameter(torch.full((), init, device=self.device))
                    for name in task_names
                }
            )
        self._trainable_parameters = list(self.model.parameters())
        if self.multitask_log_vars is not None:
            self._trainable_parameters.extend(list(self.multitask_log_vars.parameters()))
        self.optimizer = optim.Adam(self._trainable_parameters, lr=self.training.learning_rate)
        self._current_aux_scale = float(self.training.multitask_aux_scale)
        self._action_intensities = torch.as_tensor(
            [
                float(template.aggressiveness * template.size_fraction)
                for template in self.env_config.templates
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.env = MultiOptionExecutionEnv(self.env_config)
        self.league = ScenarioLeague(self.training, seed=seed)
        self.name = "OptionPPO"

    def _urgency_to_bins(self, urgency: np.ndarray) -> np.ndarray:
        return np.digitize(urgency, bins=np.array([0.20, 0.45, 0.70], dtype=np.float32)).astype(
            np.int64
        )

    def _multitask_task_names(self) -> List[str]:
        task_names: List[str] = []
        if not self.training.multitask_anchor_ppo:
            task_names.append("ppo")
        if self.training.use_module_a:
            task_names.append("module_a")
        if self.training.use_module_b:
            task_names.append("module_b")
        if self.training.use_module_c:
            task_names.append("module_c")
        return task_names

    def _combine_task_losses(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if self.multitask_log_vars is None:
            total_loss = task_losses["ppo"]
            task_metrics = {
                "ppo_task_weight": 1.0,
                "module_a_task_weight": float(self.training.module_a_weight)
                if "module_a" in task_losses
                else 0.0,
                "module_b_task_weight": float(self.training.module_b_weight)
                if "module_b" in task_losses
                else 0.0,
                "module_c_task_weight": float(self.training.module_c_weight)
                if "module_c" in task_losses
                else 0.0,
            }
            if "module_a" in task_losses:
                total_loss = total_loss + self.training.module_a_weight * task_losses["module_a"]
            if "module_b" in task_losses:
                total_loss = total_loss + self.training.module_b_weight * task_losses["module_b"]
            if "module_c" in task_losses:
                total_loss = total_loss + self.training.module_c_weight * task_losses["module_c"]
            return total_loss, task_metrics

        task_metrics: Dict[str, float] = {"multitask_aux_scale": float(self._current_aux_scale)}
        log_var_min = float(self.training.multitask_log_var_min)
        log_var_max = float(self.training.multitask_log_var_max)
        if self.training.multitask_anchor_ppo:
            total_loss = task_losses["ppo"]
            task_metrics["ppo_task_weight"] = 1.0
            aux_scale = float(self._current_aux_scale)
            for name, task_loss in task_losses.items():
                if name == "ppo":
                    continue
                log_var = torch.clamp(self.multitask_log_vars[name], log_var_min, log_var_max)
                precision = torch.exp(-log_var)
                total_loss = total_loss + aux_scale * 0.5 * (precision * task_loss + log_var)
                task_metrics[f"{name}_task_weight"] = float((aux_scale * precision).detach().cpu())
                task_metrics[f"{name}_task_log_var"] = float(log_var.detach().cpu())
            return total_loss, task_metrics

        total_loss = torch.zeros((), device=self.device)
        for name, task_loss in task_losses.items():
            log_var = torch.clamp(self.multitask_log_vars[name], log_var_min, log_var_max)
            precision = torch.exp(-log_var)
            total_loss = total_loss + 0.5 * (precision * task_loss + log_var)
            task_metrics[f"{name}_task_weight"] = float(precision.detach().cpu())
            task_metrics[f"{name}_task_log_var"] = float(log_var.detach().cpu())
        return total_loss, task_metrics

    def _clamp_multitask_log_vars(self) -> None:
        if self.multitask_log_vars is None:
            return
        log_var_min = float(self.training.multitask_log_var_min)
        log_var_max = float(self.training.multitask_log_var_max)
        with torch.no_grad():
            for parameter in self.multitask_log_vars.values():
                parameter.clamp_(log_var_min, log_var_max)

    def _update_aux_schedule(self, episode_idx: int, total_episodes: int) -> None:
        base_aux_scale = float(self.training.multitask_aux_scale)
        warmup_ratio = float(self.training.multitask_aux_warmup_ratio)
        if warmup_ratio <= 0.0 or total_episodes <= 0:
            self._current_aux_scale = base_aux_scale
            return
        progress = episode_idx / max(total_episodes, 1)
        ramp = min(max(progress / warmup_ratio, 0.0), 1.0)
        self._current_aux_scale = base_aux_scale * ramp

    def _select_actions(
        self,
        observations: List[np.ndarray],
        greedy: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_tensor = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=self.device)
        logits = self.model.actor(obs_tensor)
        values = self.model.value(obs_tensor).squeeze(-1)
        dist = Categorical(logits=logits)
        if greedy:
            if self.training.use_boltzmann_eval:
                temperature = max(float(self.training.eval_boltzmann_temperature), 1e-6)
                boltzmann_dist = Categorical(logits=logits / temperature)
                actions = boltzmann_dist.sample()
                logprobs = boltzmann_dist.log_prob(actions)
            else:
                actions = torch.argmax(logits, dim=-1)
                logprobs = dist.log_prob(actions)
        else:
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
        return (
            actions.detach().cpu().numpy(),
            logprobs.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        )

    def _public_only_obs(self, obs: torch.Tensor) -> torch.Tensor:
        public_only = obs.clone()
        public_only[..., : self.env_config.local_feature_dim] = 0.0
        return public_only

    def _apply_public_trace_intervention(
        self,
        observations: List[np.ndarray],
        corrupt_public_trace: bool,
    ) -> List[np.ndarray]:
        if not corrupt_public_trace:
            return observations
        return self.env.apply_public_trace_intervention(observations)

    def _corrupt_public_trace_tensor(
        self,
        obs: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"expected [batch, obs_dim] tensor, got shape {tuple(obs.shape)}")
        corrupted = obs.clone()
        public = corrupted[:, self.env_config.local_feature_dim :].reshape(
            -1,
            self.env_config.public_tape_window,
            self.env_config.num_contracts,
            3,
        )
        batch_size = public.shape[0]
        teammate_mask = torch.ones(
            (batch_size, self.env_config.num_contracts),
            dtype=torch.bool,
            device=obs.device,
        )
        teammate_mask[
            torch.arange(batch_size, device=obs.device),
            agent_indices.reshape(-1).long(),
        ] = False
        public.masked_fill_(teammate_mask[:, None, :, None], 0.0)
        corrupted[:, self.env_config.local_feature_dim :] = public.reshape(
            -1,
            self.env_config.public_tape_dim,
        )
        return corrupted

    def _symmetric_policy_consistency_loss(
        self,
        clean_logits: torch.Tensor,
        corrupted_logits: torch.Tensor,
    ) -> torch.Tensor:
        clean_log_probs = F.log_softmax(clean_logits, dim=-1)
        corrupted_log_probs = F.log_softmax(corrupted_logits, dim=-1)
        clean_probs = clean_log_probs.exp()
        corrupted_probs = corrupted_log_probs.exp()
        clean_to_corrupted = torch.sum(
            clean_probs * (clean_log_probs - corrupted_log_probs),
            dim=-1,
        ).mean()
        corrupted_to_clean = torch.sum(
            corrupted_probs * (corrupted_log_probs - clean_log_probs),
            dim=-1,
        ).mean()
        return 0.5 * (clean_to_corrupted + corrupted_to_clean)

    def _info_nce_loss(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if queries.shape[0] <= 1:
            return torch.zeros((), device=self.device), 0.0
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        logits = torch.matmul(queries, keys.transpose(0, 1))
        logits = logits / max(self.training.module_b_temperature, 1e-6)
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = F.cross_entropy(logits, labels)
        mi_lower_bound = max(0.0, float(np.log(logits.shape[0]) - float(loss.detach().cpu())))
        return loss, mi_lower_bound

    def _collect_episode(
        self,
        training: bool = True,
        corrupt_public_trace: bool = False,
        sample_league_scenario: bool = False,
    ) -> tuple[EpisodeRollout, Dict[str, float]]:
        scenario = (
            self.league.sample()
            if ((training or sample_league_scenario) and self.training.use_module_c)
            else None
        )
        observations = self.env.reset(seed=self.seed + np.random.randint(0, 100000), scenario=scenario)
        episode_obs: List[np.ndarray] = []
        episode_clean_obs: List[np.ndarray] = []
        episode_next_obs: List[np.ndarray] = []
        episode_privileged: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []
        episode_logprobs: List[np.ndarray] = []
        episode_rewards: List[np.ndarray] = []
        episode_dones: List[np.ndarray] = []
        episode_values: List[np.ndarray] = []
        episode_belief_targets: List[np.ndarray] = []
        episode_sender_targets: List[np.ndarray] = []
        episode_receiver_targets: List[np.ndarray] = []
        episode_module_c_targets: List[np.ndarray] = []
        episode_agent_indices: List[np.ndarray] = []
        episode_trace_dropout = (
            training
            and self.training.trace_dropout_prob > 0.0
            and float(np.random.random()) < float(self.training.trace_dropout_prob)
        )

        done = False
        while not done:
            clean_policy_observations = [np.array(observation, copy=True) for observation in observations]
            policy_observations = self._apply_public_trace_intervention(
                observations,
                corrupt_public_trace=(corrupt_public_trace or episode_trace_dropout),
            )
            actions, logprobs, values = self._select_actions(
                policy_observations,
                greedy=not training,
            )
            next_observations, reward, done, info = self.env.step(actions.tolist())
            policy_next_observations = self._apply_public_trace_intervention(
                next_observations,
                corrupt_public_trace=(corrupt_public_trace or episode_trace_dropout),
            )
            urgency = np.asarray(info["urgency_targets"], dtype=np.float32)
            sender_targets = self._urgency_to_bins(urgency)
            teammate_urgency = np.array(
                [
                    float(np.mean(np.delete(urgency, idx))) if len(urgency) > 1 else urgency[idx]
                    for idx in range(len(urgency))
                ],
                dtype=np.float32,
            )
            receiver_targets = self._urgency_to_bins(teammate_urgency)

            minimax_bonus = _compute_shifted_minimax_bonus(
                self.training,
                info,
                done=done,
                training=training,
            )
            shaped_reward = reward + minimax_bonus

            agent_reward = np.full(self.env_config.num_contracts, shaped_reward, dtype=np.float32)
            agent_done = np.full(self.env_config.num_contracts, float(done), dtype=np.float32)
            privileged = np.repeat(
                np.asarray(info["privileged_state"], dtype=np.float32)[None, :],
                self.env_config.num_contracts,
                axis=0,
            )
            belief_targets = np.repeat(
                np.asarray(info["belief_target"], dtype=np.float32)[None, :],
                self.env_config.num_contracts,
                axis=0,
            )
            module_c_target = _compute_module_c_aux_target(self.training, info)
            module_c_targets = np.full(
                self.env_config.num_contracts,
                module_c_target,
                dtype=np.float32,
            )

            episode_obs.append(np.stack(policy_observations).astype(np.float32))
            episode_clean_obs.append(np.stack(clean_policy_observations).astype(np.float32))
            episode_next_obs.append(np.stack(policy_next_observations).astype(np.float32))
            episode_privileged.append(privileged)
            episode_actions.append(actions.astype(np.int64))
            episode_logprobs.append(logprobs.astype(np.float32))
            episode_rewards.append(agent_reward)
            episode_dones.append(agent_done)
            episode_values.append(values.astype(np.float32))
            episode_belief_targets.append(belief_targets.astype(np.float32))
            episode_sender_targets.append(sender_targets.astype(np.int64))
            episode_receiver_targets.append(receiver_targets.astype(np.int64))
            episode_module_c_targets.append(module_c_targets)
            episode_agent_indices.append(
                np.arange(self.env_config.num_contracts, dtype=np.int64)
            )
            observations = next_observations

        rewards = np.stack(episode_rewards)
        dones = np.stack(episode_dones)
        values = np.stack(episode_values)
        returns, advantages = self._compute_gae(rewards, dones, values)

        rollout = EpisodeRollout(
            obs=np.stack(episode_obs),
            clean_obs=np.stack(episode_clean_obs),
            next_obs=np.stack(episode_next_obs),
            privileged=np.stack(episode_privileged),
            actions=np.stack(episode_actions),
            logprobs=np.stack(episode_logprobs),
            rewards=rewards,
            dones=dones,
            values=values,
            returns=returns,
            advantages=advantages,
            belief_targets=np.stack(episode_belief_targets),
            sender_targets=np.stack(episode_sender_targets),
            receiver_targets=np.stack(episode_receiver_targets),
            module_c_targets=np.stack(episode_module_c_targets),
            agent_indices=np.stack(episode_agent_indices),
        )
        return rollout, self.env.get_metrics()

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)
        next_values = np.zeros(rewards.shape[1], dtype=np.float32)
        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.training.gamma * next_values * mask - values[t]
            gae = delta + self.training.gamma * self.training.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_values = values[t]
        return returns, advantages

    def _flatten_rollout(self, rollout: EpisodeRollout) -> Dict[str, torch.Tensor]:
        flat: Dict[str, torch.Tensor] = {}
        for field_name in rollout.__dataclass_fields__:
            array = getattr(rollout, field_name)
            tensor = torch.as_tensor(array.reshape(-1, *array.shape[2:]), device=self.device)
            if field_name in {"actions", "sender_targets", "receiver_targets", "agent_indices"}:
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            flat[field_name] = tensor
        return flat

    def _update_model(self, rollout: EpisodeRollout) -> Dict[str, float]:
        flat = self._flatten_rollout(rollout)
        advantages = flat["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_logprobs = flat["logprobs"]
        action_tensor = flat["actions"].squeeze(-1) if flat["actions"].ndim > 1 else flat["actions"]
        returns = flat["returns"]
        metrics: Dict[str, float] = {}

        for _ in range(self.training.update_epochs):
            logits = self.model.actor(flat["obs"])
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()
            values = self.model.value(flat["obs"]).squeeze(-1)

            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.training.clip_eps, 1.0 + self.training.clip_eps
            ) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            ppo_loss = actor_loss + self.training.value_coef * value_loss - self.training.entropy_coef * entropy

            completion_regularizer_loss = torch.zeros((), device=self.device)
            completion_regularizer_penalty = torch.zeros((), device=self.device)
            if self.training.completion_regularizer_weight > 0.0:
                action_probs = torch.softmax(logits, dim=-1)
                expected_intensity = torch.sum(
                    action_probs * self._action_intensities.unsqueeze(0),
                    dim=-1,
                )
                target_intensity = flat["sender_targets"].float() / 3.0
                completion_regularizer_loss = F.mse_loss(expected_intensity, target_intensity)
                completion_regularizer_penalty = (
                    float(self.training.completion_regularizer_weight)
                    * completion_regularizer_loss
                )

            module_a_loss = torch.tensor(0.0, device=self.device)
            if self.training.use_module_a:
                teacher_values = self.model.privileged_value(flat["privileged"]).squeeze(-1)
                teacher_loss = F.mse_loss(teacher_values, returns)
                distill_loss = F.mse_loss(values, teacher_values.detach())
                belief_preds = torch.sigmoid(self.model.belief(flat["obs"]))
                belief_loss = F.mse_loss(belief_preds, flat["belief_targets"])
                module_a_loss = teacher_loss + distill_loss + belief_loss

            module_b_loss = torch.tensor(0.0, device=self.device)
            sender_accuracy = 0.0
            receiver_accuracy = 0.0
            action_public_mi = 0.0
            intent_public_mi = 0.0
            if self.training.use_module_b:
                sender_logits, _ = self.model.coordination(flat["obs"])
                _, receiver_logits = self.model.coordination(self._public_only_obs(flat["next_obs"]))
                sender_loss = F.cross_entropy(sender_logits, flat["sender_targets"])
                receiver_loss = F.cross_entropy(receiver_logits, flat["receiver_targets"])
                sender_latent, _ = self.model.coordination_latents(flat["obs"])
                _, receiver_latent = self.model.coordination_latents(
                    self._public_only_obs(flat["next_obs"])
                )
                action_latent = self.model.embed_actions(action_tensor)
                action_mi_loss, action_public_mi = self._info_nce_loss(receiver_latent, action_latent)
                intent_mi_loss, intent_public_mi = self._info_nce_loss(receiver_latent, sender_latent)
                module_b_loss = sender_loss + receiver_loss + self.training.module_b_mi_weight * (
                    action_mi_loss + intent_mi_loss
                )
                sender_accuracy = float(
                    (sender_logits.argmax(dim=-1) == flat["sender_targets"])
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
                receiver_accuracy = float(
                    (receiver_logits.argmax(dim=-1) == flat["receiver_targets"])
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )

            module_c_loss = torch.tensor(0.0, device=self.device)
            module_c_mae = 0.0
            module_c_prediction_mean = 0.0
            module_c_target_mean = 0.0
            if self.training.use_module_c:
                module_c_preds = torch.sigmoid(self.model.robustness(flat["obs"]).squeeze(-1))
                module_c_loss = F.mse_loss(module_c_preds, flat["module_c_targets"])
                module_c_mae = float(
                    F.l1_loss(module_c_preds, flat["module_c_targets"]).detach().cpu()
                )
                module_c_prediction_mean = float(module_c_preds.mean().detach().cpu())
                module_c_target_mean = float(flat["module_c_targets"].mean().detach().cpu())

            task_losses = {"ppo": ppo_loss}
            if self.training.use_module_a:
                task_losses["module_a"] = module_a_loss
            if self.training.use_module_b:
                task_losses["module_b"] = module_b_loss
            if self.training.use_module_c:
                task_losses["module_c"] = module_c_loss
            total_loss, task_weight_metrics = self._combine_task_losses(task_losses)

            trace_consistency_loss = torch.zeros((), device=self.device)
            trace_consistency_penalty = torch.zeros((), device=self.device)
            if self.training.trace_consistency_weight > 0.0:
                clean_obs = flat["clean_obs"]
                corrupted_clean_obs = self._corrupt_public_trace_tensor(
                    clean_obs,
                    flat["agent_indices"],
                )
                clean_reference_logits = self.model.actor(clean_obs)
                clean_reference_values = self.model.value(clean_obs).squeeze(-1)
                corrupted_reference_logits = self.model.actor(corrupted_clean_obs)
                corrupted_reference_values = self.model.value(corrupted_clean_obs).squeeze(-1)
                trace_policy_loss = self._symmetric_policy_consistency_loss(
                    clean_reference_logits,
                    corrupted_reference_logits,
                )
                trace_value_loss = F.mse_loss(
                    clean_reference_values,
                    corrupted_reference_values,
                )
                trace_consistency_loss = 0.5 * trace_policy_loss + 0.5 * trace_value_loss
                trace_consistency_penalty = (
                    float(self.training.trace_consistency_weight)
                    * trace_consistency_loss
                )

            total_loss = total_loss + completion_regularizer_penalty + trace_consistency_penalty

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_parameters, 1.0)
            self.optimizer.step()
            self._clamp_multitask_log_vars()

            metrics = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "module_a_loss": float(module_a_loss.detach().cpu()),
                "module_b_loss": float(module_b_loss.detach().cpu()),
                "module_c_loss": float(module_c_loss.detach().cpu()),
                "module_c_mae": float(module_c_mae),
                "module_c_prediction_mean": float(module_c_prediction_mean),
                "module_c_target_mean": float(module_c_target_mean),
                "completion_regularizer_loss": float(completion_regularizer_loss.detach().cpu()),
                "completion_regularizer_penalty": float(
                    completion_regularizer_penalty.detach().cpu()
                ),
                "trace_consistency_loss": float(trace_consistency_loss.detach().cpu()),
                "trace_consistency_penalty": float(
                    trace_consistency_penalty.detach().cpu()
                ),
                "sender_accuracy": float(sender_accuracy),
                "receiver_accuracy": float(receiver_accuracy),
                "action_public_mi_lb": float(action_public_mi),
                "intent_public_mi_lb": float(intent_public_mi),
                "total_loss": float(total_loss.detach().cpu()),
                **task_weight_metrics,
            }
        return metrics

    def train(self, episodes: Optional[int] = None) -> List[Dict[str, float]]:
        total_episodes = episodes or self.training.episodes
        history: List[Dict[str, float]] = []
        for episode_idx in range(1, total_episodes + 1):
            self._update_aux_schedule(episode_idx=episode_idx, total_episodes=total_episodes)
            rollout, env_metrics = self._collect_episode(training=True)
            train_metrics = self._update_model(rollout)
            merged = {"episode": episode_idx, **env_metrics, **train_metrics}
            history.append(merged)
            if self.training.use_module_c:
                self.league.record(episode_idx, env_metrics)
            if episode_idx % self.training.log_interval == 0:
                print(
                    f"[{self.name}] episode={episode_idx} "
                    f"shortfall={env_metrics['implementation_shortfall']:.3f} "
                    f"completion={env_metrics['completion_rate']:.3f} "
                    f"reward={env_metrics['reward']:.3f}"
                )
        return history

    def _coordination_diagnostics(self, rollout: EpisodeRollout) -> Dict[str, float]:
        if not self.training.use_module_b:
            return {}
        flat = self._flatten_rollout(rollout)
        action_tensor = flat["actions"].squeeze(-1) if flat["actions"].ndim > 1 else flat["actions"]
        with torch.no_grad():
            sender_logits, _ = self.model.coordination(flat["obs"])
            _, receiver_logits = self.model.coordination(self._public_only_obs(flat["next_obs"]))
            sender_latent, _ = self.model.coordination_latents(flat["obs"])
            _, receiver_latent = self.model.coordination_latents(self._public_only_obs(flat["next_obs"]))
            action_latent = self.model.embed_actions(action_tensor)
            _, action_public_mi = self._info_nce_loss(receiver_latent, action_latent)
            _, intent_public_mi = self._info_nce_loss(receiver_latent, sender_latent)
            sender_accuracy = float(
                (sender_logits.argmax(dim=-1) == flat["sender_targets"]).float().mean().cpu()
            )
            receiver_accuracy = float(
                (receiver_logits.argmax(dim=-1) == flat["receiver_targets"]).float().mean().cpu()
            )
        return {
            "sender_accuracy": sender_accuracy,
            "receiver_accuracy": receiver_accuracy,
            "action_public_mi_lb": float(action_public_mi),
            "intent_public_mi_lb": float(intent_public_mi),
        }

    def _module_c_diagnostics(self, rollout: EpisodeRollout) -> Dict[str, float]:
        if not self.training.use_module_c:
            return {}
        flat = self._flatten_rollout(rollout)
        with torch.no_grad():
            module_c_preds = torch.sigmoid(self.model.robustness(flat["obs"]).squeeze(-1))
        return {
            "module_c_mae": float(F.l1_loss(module_c_preds, flat["module_c_targets"]).cpu()),
            "module_c_prediction_mean": float(module_c_preds.mean().cpu()),
            "module_c_target_mean": float(flat["module_c_targets"].mean().cpu()),
        }

    def evaluate(
        self,
        episodes: Optional[int] = None,
        corrupt_public_trace: bool = False,
    ) -> Dict[str, float]:
        total = episodes or self.training.eval_episodes
        records: List[Dict[str, float]] = []
        for _ in range(total):
            rollout, metrics = self._collect_episode(
                training=False,
                corrupt_public_trace=corrupt_public_trace,
            )
            module_c_rollout = rollout
            if self.training.use_module_c:
                module_c_rollout, _ = self._collect_episode(
                    training=False,
                    corrupt_public_trace=corrupt_public_trace,
                    sample_league_scenario=True,
                )
            diagnostics = {
                **self._coordination_diagnostics(rollout),
                **self._module_c_diagnostics(module_c_rollout),
            }
            metrics = {**metrics, **diagnostics}
            records.append(metrics)
        return {
            key: float(np.mean([record[key] for record in records]))
            for key in records[0]
        }


class IndependentPPOSolver(OptionPPOSolver):
    def __init__(
        self,
        env_config: Optional[OptionExecutionConfig] = None,
        training_config: Optional[OptionTrainingConfig] = None,
        device: Optional[str] = None,
        seed: int = 7,
    ) -> None:
        training = training_config or OptionTrainingConfig()
        training.use_module_a = False
        training.use_module_b = False
        training.use_module_c = False
        super().__init__(env_config=env_config, training_config=training, device=device, seed=seed)
        self.name = "Independent PPO"


class ATDecOptionSolver(OptionPPOSolver):
    def __init__(
        self,
        env_config: Optional[OptionExecutionConfig] = None,
        training_config: Optional[OptionTrainingConfig] = None,
        device: Optional[str] = None,
        seed: int = 7,
    ) -> None:
        training = training_config or OptionTrainingConfig()
        training.use_module_a = True
        training.use_module_b = True
        training.use_module_c = True
        super().__init__(env_config=env_config, training_config=training, device=device, seed=seed)
        self.name = "AT-Dec Option Solver"
