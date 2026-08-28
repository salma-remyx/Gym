# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test training-sample emission at the token-capture boundary."""

import asyncio

from nemo_gym.token_id_capture import (
    TokenCaptureSnapshot,
    build_training_sample,
    dedup_identical_samples,
    group_advantages,
    trajectories_from_source,
)
from nemo_gym.token_id_capture.records import TokenEntry


def _entry(mcid, prompt, gen, lp=None):
    return TokenEntry(
        rollout_id="t0-r0",
        model_call_id=mcid,
        model="m",
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=lp if lp is not None else [-0.1] * len(gen),
    )


# An append-only rollout of two calls.
# The full contiguous sequence is [1, 2, 3, 10, 11, 4, 12].
CALL1 = _entry("c1", [1, 2, 3], [10, 11])
CALL2 = _entry("c2", [1, 2, 3, 10, 11, 4], [12])
APPEND_ONLY = [CALL1, CALL2]


def _source(entries):
    class Source:
        async def freeze(self, rollout_id):
            return TokenCaptureSnapshot(
                rollout_id=rollout_id,
                entries=tuple(entries),
                incomplete=False,
                snapshot_id="snapshot-1",
                version=1,
            )

        async def drop(self, rollout_id, *, snapshot_id, version):
            return True

        async def close(self):
            return None

    return Source()


def test_consumer_attaches_a_training_ready_sample():
    built = asyncio.run(trajectories_from_source("t0-r0", _source(APPEND_ONLY)))

    assert built["mask_sample"] is False
    sample = built["training_sample"]
    assert sample is not None
    # The flattened sequence is the first prompt plus every generation.
    assert sample["input_ids"] == [1, 2, 3, 10, 11, 4, 12]
    # Prompt positions are context; generated positions are trained on.
    assert sample["loss_mask"] == [0, 0, 0, 1, 1, 0, 1]
    assert sample["log_probs"] == [0.0, 0.0, 0.0, -0.1, -0.1, 0.0, -0.1]
    # Rollout-level token-mean: each generated token weighs 1/3 so the
    # rollout's contribution is independent of its length.
    assert sample["n_generated_tokens"] == 3
    assert sample["loss_weights"] == [0.0, 0.0, 0.0, 1 / 3, 1 / 3, 0.0, 1 / 3]
    assert abs(sum(sample["loss_weights"]) - 1.0) < 1e-9


def test_failed_build_has_no_training_sample():
    # A masked build still reports the key so consumers have one shape.
    built = asyncio.run(trajectories_from_source("t0-r0", _source([_entry("c1", [1], [])])))

    assert built["mask_sample"] is True
    assert built["training_sample"] is None


def test_build_training_sample_rejects_a_tokenless_response():
    assert build_training_sample("t0-r0", {"output": [{"type": "message"}]}) is None
    assert build_training_sample("t0-r0", {"output": []}) is None


def test_group_advantages_use_unmasked_group_mean_baselines():
    samples = [
        {"reward": 1.0, "group_id": "g", "masked": False},
        {"reward": 0.0, "group_id": "g", "masked": False},
        # A masked sample is excluded from the baseline and from the loss.
        {"reward": 100.0, "group_id": "g", "masked": True},
        {"reward": 1.0, "group_id": "other", "masked": False},
    ]

    assert group_advantages(samples) == [0.5, -0.5, 0.0, 0.0]


def test_group_advantages_tolerate_a_fully_masked_group():
    samples = [{"reward": 1.0, "group_id": "g", "masked": True}]

    assert group_advantages(samples) == [0.0]


def test_dedup_drops_only_identical_sequences():
    duplicate_a = {"input_ids": [1, 2, 3], "rollout_id": "a"}
    duplicate_b = {"input_ids": [1, 2, 3], "rollout_id": "b"}
    # Same prompt, different generation: an intentional group member.
    sibling = {"input_ids": [1, 2, 9], "rollout_id": "c"}

    kept, dropped = dedup_identical_samples([duplicate_a, duplicate_b, sibling])

    assert kept == [duplicate_a, sibling]
    assert dropped == [duplicate_b]


def test_dedup_keeps_samples_without_token_ids():
    samples = [{"rollout_id": "a"}, {"rollout_id": "b"}]

    kept, dropped = dedup_identical_samples(samples)

    assert kept == samples
    assert dropped == []
