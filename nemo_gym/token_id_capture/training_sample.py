# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Emit training-ready samples from a rebuilt rollout.

Adapted from Agent Lightning v1.0 (arXiv:2608.17528), "Towards Harnessed
Agentic RL". In harnessed agentic RL the trainer observes only
request-response pairs, so harness-side semantics that a downstream RL
trainer cannot recover from flat samples must be attached at the capture
boundary, right after chain building:

- Rollout-level token-mean loss weights (paper Eq. 16). Every generated
  token in a rollout carries weight ``1 / N`` where ``N`` is the
  rollout's generated-token count. Long and short rollouts then
  contribute equally to the policy-gradient loss.
- Rollout-level advantage grouping (paper section 2.2). Group baselines
  are computed from unmasked samples only. Masked samples (auxiliary
  calls, quarantined retries, broken terminal chains) are excluded from
  baselines and receive zero advantage.
- Identical-sequence retry dedup (paper section 3.2). Within a batch,
  samples with an identical flattened token sequence are duplicate
  deliveries of one rollout; only the first is kept.

The per-rollout sample consumes the projected Responses payload from
``nemo_gym.token_id_capture.builder.project_main_chain_response``.
"""

from __future__ import annotations

TRAINING_SAMPLE_KEY = "training_sample"


def flatten_projected_tokens(response: dict) -> tuple[list[int], list[int], list[float]]:
    """Flatten a projected response into contiguous token, mask, and log-prob arrays.

    Each token-bearing item's prompt extends all preceding tokens, so an
    item contributes the prompt tokens not yet emitted (interstitial tool
    output or user turns) followed by its generation. Prompt positions
    have a loss mask of 0. Generated positions have a loss mask of 1 and
    keep their log probs. Raise ``ValueError`` when the response is not
    prefix-contiguous.
    """
    token_ids: list[int] = []
    loss_mask: list[int] = []
    log_probs: list[float] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("generation_token_ids") is None:
            continue
        prompt = list(item.get("prompt_token_ids") or [])
        if prompt[: len(token_ids)] != token_ids:
            raise ValueError("response is not prefix-contiguous")
        new_prompt = prompt[len(token_ids) :]
        generation = list(item["generation_token_ids"])
        token_ids.extend(new_prompt)
        loss_mask.extend([0] * len(new_prompt))
        log_probs.extend([0.0] * len(new_prompt))
        token_ids.extend(generation)
        loss_mask.extend([1] * len(generation))
        log_probs.extend(list(item.get("generation_log_probs") or [0.0] * len(generation)))
    return token_ids, loss_mask, log_probs


def build_training_sample(rollout_id: str, response: dict) -> dict | None:
    """Attach harness-side training semantics to a rebuilt rollout response.

    Return ``None`` when the response carries no generated tokens.
    The sample holds the contiguous token sequence, its loss mask, the
    captured log probabilities, and rollout-level token-mean loss
    weights. The reward and advantage group are rollout-batch concerns;
    the caller assigns them per rollout before batch assembly.
    """
    token_ids, loss_mask, log_probs = flatten_projected_tokens(response)
    n_generated = sum(loss_mask)
    if n_generated == 0:
        return None
    # Rollout-level token-mean (paper Eq. 16): the trainer multiplies
    # each token's loss by this weight, so a rollout's contribution does
    # not scale with its length.
    token_weight = 1.0 / n_generated
    return {
        "rollout_id": rollout_id,
        "input_ids": token_ids,
        "loss_mask": loss_mask,
        "log_probs": log_probs,
        "loss_weights": [token_weight if mask else 0.0 for mask in loss_mask],
        "n_generated_tokens": n_generated,
    }


def group_advantages(samples: list[dict]) -> list[float]:
    """Compute rollout-level advantages against group-mean baselines.

    Each sample needs ``reward`` (float), ``group_id`` (str), and
    ``masked`` (bool). Rollouts of one task share a group id. A masked
    sample is excluded from its group's baseline and receives zero
    advantage (paper section 2.2): calls the verifier never scored must
    not pull the baseline toward themselves.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sample in samples:
        if sample.get("masked"):
            continue
        group = str(sample.get("group_id", ""))
        totals[group] = totals.get(group, 0.0) + float(sample.get("reward", 0.0))
        counts[group] = counts.get(group, 0) + 1
    advantages: list[float] = []
    for sample in samples:
        if sample.get("masked"):
            advantages.append(0.0)
            continue
        group = str(sample.get("group_id", ""))
        baseline = totals[group] / counts[group] if counts.get(group) else 0.0
        advantages.append(float(sample.get("reward", 0.0)) - baseline)
    return advantages


def dedup_identical_samples(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop samples whose flattened token sequence already appears in the batch.

    A retried delivery can emit the same rollout twice. Identical
    sequences carry no additional gradient signal, so only the first
    occurrence is kept (paper section 3.2). Samples that share a prompt
    but differ in generation are intentional group members and are kept.
    Return the kept samples and the dropped duplicates.
    """
    seen: set[tuple] = set()
    kept: list[dict] = []
    dropped: list[dict] = []
    for sample in samples:
        key = tuple(sample.get("input_ids") or [])
        if key and key in seen:
            dropped.append(sample)
            continue
        seen.add(key)
        kept.append(sample)
    return kept, dropped
