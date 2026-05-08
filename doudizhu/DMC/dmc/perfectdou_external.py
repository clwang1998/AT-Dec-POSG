"""External PerfectDou opponent wrapper for mainline DMC training.

This module keeps PerfectDou out of the learner graph. It only exposes a
greedy action-selection wrapper that can be plugged into the league opponent
sampler as an external sparring policy.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_PERFECTDOU_ROOT = PROJECT_ROOT.parent / "PerfectDou-official"
DEFAULT_OFFICIAL_PERFECTDOU_ROOT = Path(
    os.environ.get("PERFECTDOU_REPO_ROOT", str(DEFAULT_REPO_PERFECTDOU_ROOT))
)
DEFAULT_OFFICIAL_PERFECTDOU_DIR = Path(
    os.environ.get(
        "PERFECTDOU_DIR",
        str(DEFAULT_OFFICIAL_PERFECTDOU_ROOT / "perfectdou" / "model" / "perfectdou"),
    )
)

PLAY_POSITIONS = ("landlord", "landlord_up", "landlord_down")
PERFECTDOU_CARD_MAP = {
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
    "2": 17,
    "B": 20,
    "R": 30,
}


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


def _perfectdou_infoset_adapter(infoset):
    adapted = deepcopy(infoset)
    action_seq = getattr(adapted, "card_play_action_seq", None)
    if action_seq:
        normalized = []
        for item in action_seq:
            if isinstance(item, tuple) and len(item) == 2:
                normalized.append(list(item[1]))
            else:
                normalized.append(list(item))
        adapted.card_play_action_seq = normalized
    return adapted


class PerfectDouExternalOpponent:
    """Greedy legal-action opponent built from official PerfectDou ONNX files."""

    def __init__(
        self,
        perfectdou_dir: Path = DEFAULT_OFFICIAL_PERFECTDOU_DIR,
        repo_root: Path = DEFAULT_OFFICIAL_PERFECTDOU_ROOT,
        projection: str = "logsumexp",
    ):
        self.perfectdou_dir = Path(perfectdou_dir).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.projection = projection
        self._initialized = False
        self._sessions: Dict[str, object] = {}

        self.action_space = json.loads(
            (self.repo_root / "action_space.json").read_text(encoding="utf-8")
        )
        self.specific_map = json.loads(
            (self.repo_root / "specific_map.json").read_text(encoding="utf-8")
        )

    def _ensure_ready(self) -> None:
        if self._initialized:
            return
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        try:
            with _temporary_cwd(self.repo_root):
                import onnxruntime as ort  # type: ignore
                from perfectdou.env.encode import (  # type: ignore
                    _decode_action,
                    encode_obs_landlord,
                    encode_obs_peasant,
                )
        except Exception as exc:  # pragma: no cover - runtime-dependent
            raise RuntimeError(
                "PerfectDou external opponent requires onnxruntime and perfectdou.env.encode."
            ) from exc
        self.ort = ort
        self._decode_action = _decode_action
        self._encode_obs_landlord = encode_obs_landlord
        self._encode_obs_peasant = encode_obs_peasant
        for position in PLAY_POSITIONS:
            self._sessions[position] = ort.InferenceSession(
                str(self.perfectdou_dir / f"{position}.onnx"),
                providers=["CPUExecutionProvider"],
            )
        self._initialized = True

    def _abstract_labels_for_action(self, action_str: str) -> Tuple[str, ...]:
        if action_str == "pass":
            return ("pass",)
        labels = self.specific_map.get(action_str)
        if labels:
            return tuple(labels)
        return (action_str,)

    def _env_action(self, action_repr) -> List[int]:
        if action_repr == "pass":
            return []
        return [PERFECTDOU_CARD_MAP[card] for card in action_repr]

    def _project_scores(self, logits: np.ndarray, action_strings: Sequence[str]) -> torch.Tensor:
        projected = []
        for action_str in action_strings:
            label_ids = [
                int(self.action_space[label])
                for label in self._abstract_labels_for_action(action_str)
                if label in self.action_space
            ]
            if not label_ids:
                projected.append(-1e9)
                continue
            label_logits = torch.tensor(logits[label_ids], dtype=torch.float32)
            if self.projection == "max":
                projected.append(float(label_logits.max().item()))
            else:
                projected.append(float(torch.logsumexp(label_logits, dim=0).item()))
        return torch.tensor(projected, dtype=torch.float32)

    def _fallback_action_index(self, logits: np.ndarray, obs: dict, infoset) -> int:
        action_id = int(np.argmax(logits))
        decoded = self._decode_action(action_id, obs["current_hand"], obs["actions"])
        env_action = self._env_action(decoded)
        for index, legal_action in enumerate(infoset.legal_actions):
            if list(legal_action) == env_action:
                return index
        return 0

    def action_index(self, infoset) -> int:
        self._ensure_ready()
        infoset = _perfectdou_infoset_adapter(infoset)
        position = infoset.player_position
        if position == "landlord":
            obs = self._encode_obs_landlord(infoset)
        else:
            obs = self._encode_obs_peasant(infoset)
        session = self._sessions[position]
        input_name = session.get_inputs()[0].name
        input_data = np.concatenate(
            [obs["x_no_action"].flatten(), obs["legal_actions_arr"].flatten()]
        ).reshape(1, -1)
        logits = session.run(["action_logit"], {input_name: input_data})[0].reshape(-1)
        legal_scores = self._project_scores(logits, obs["actions"])
        if torch.isfinite(legal_scores).any():
            return int(torch.argmax(legal_scores).item())
        return self._fallback_action_index(logits, obs, infoset)

    def act_from_infoset(self, position, infoset, flags=None):
        if infoset is None:
            raise RuntimeError("PerfectDou external opponent needs the current infoset.")
        action_index = self.action_index(infoset)
        return {
            "action": torch.tensor(action_index),
            "max_value": torch.tensor(0.0),
        }
