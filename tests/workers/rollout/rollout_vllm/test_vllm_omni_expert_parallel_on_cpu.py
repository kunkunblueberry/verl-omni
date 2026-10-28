# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for rollout expert-parallel honor-or-reject semantics (RFC #820 S1).

The contract: ``actor_rollout_ref.rollout.expert_parallel_size`` either configures
expert parallelism or fails loudly. vLLM-Omni takes EP as a per-stage flag over the
ranks a stage already spans, so the rollout forwards it only for pipelines that opt
a stage in, and refuses the knob everywhere else.
"""

import types

import pytest
import yaml

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_omni.omni_rollout_adapter import Qwen3OmniRolloutAdapter
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

QWEN3_OMNI_PIPELINE = "qwen3_omni_moe"


def _ar_strategy(*, ep=1, tp=2, dp=1, pipeline_name=QWEN3_OMNI_PIPELINE, pipeline_mode="thinker_only"):
    config = types.SimpleNamespace(
        max_model_len=None,
        prompt_length=512,
        response_length=256,
        expert_parallel_size=ep,
        tensor_model_parallel_size=tp,
        data_parallel_size=dp,
        engine_kwargs={"vllm_omni": {"pipeline_name": pipeline_name, "pipeline_mode": pipeline_mode}},
    )
    return ARStrategy(types.SimpleNamespace(config=config))


def _diffusion_strategy(*, ep=1, parallel_config=None, flat_flag=False):
    omni_kwargs: dict = {}
    if parallel_config is not None:
        omni_kwargs["parallel_config"] = parallel_config
    if flat_flag:
        omni_kwargs["enable_expert_parallel"] = True
    config = types.SimpleNamespace(expert_parallel_size=ep, engine_kwargs={"vllm_omni": omni_kwargs})
    return DiffusionStrategy(types.SimpleNamespace(config=config))


# ---------------------------------------------------------------------------
# Qwen3-Omni opt-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline_mode", ["thinker_only", "thinker_talker", "full"])
def test_qwen3_omni_opts_its_policy_stage_in(pipeline_mode):
    """The Thinker is the policy stage and the only stage declared EP-capable."""
    assert Qwen3OmniRolloutAdapter.expert_parallel_stage_ids(pipeline_mode) == frozenset({0})
    assert Qwen3OmniRolloutAdapter.policy_stage_id(pipeline_mode=pipeline_mode) == 0


def test_qwen3_omni_is_listed_as_an_ep_pipeline():
    """The refusal message names the supported combinations from the registry."""
    assert QWEN3_OMNI_PIPELINE in OmniRolloutPipelineBase.expert_parallel_pipelines()


def test_qwen3_omni_rejects_an_unknown_pipeline_mode():
    with pytest.raises(ValueError, match="Unknown pipeline_mode"):
        Qwen3OmniRolloutAdapter.expert_parallel_stage_ids("thinker_only_typo")


# ---------------------------------------------------------------------------
# AR rollout honor-or-reject
# ---------------------------------------------------------------------------


def test_ep_one_is_admitted_without_touching_the_deploy_config():
    strategy = _ar_strategy(ep=1, tp=4)
    strategy.validate_configs()
    assert strategy._expert_parallel_stage_ids == frozenset()


def test_matching_ep_is_honored_on_the_policy_stage():
    strategy = _ar_strategy(ep=2, tp=2, dp=1)
    strategy.validate_configs()
    assert strategy._expert_parallel_stage_ids == frozenset({0})


def test_ep_must_equal_tp_times_dp():
    """vLLM derives the EP degree from TP x DP; a mismatched size cannot be honored."""
    strategy = _ar_strategy(ep=4, tp=2, dp=1)
    with pytest.raises(ValueError, match=r"tensor_model_parallel_size \* data_parallel_size"):
        strategy.validate_configs()


def test_ep_with_dp_greater_than_one_is_refused():
    """The vllm-omni rollout realizes DP as separate TP-sized replicas.

    The generated deploy config has no data_parallel_size field, so a stage can
    only shard experts over its own TP ranks: EP=TP*DP with DP>1 would silently
    deliver EP=TP.
    """
    strategy = _ar_strategy(ep=4, tp=2, dp=2)
    with pytest.raises(ValueError, match=r"data_parallel_size=2 is not wired"):
        strategy.validate_configs()


def test_ep_is_honored_when_reaching_the_replica_without_driver_validation(monkeypatch):
    """preprocess_engine_kwargs must not silently drop EP if validate was skipped.

    ``_write_deploy_config`` reads the resolved stage set from ``self``, so a
    caller that reaches engine creation without ``validate_configs`` still gets
    the flag written instead of a silently EP-less deploy config. The EP=2 case
    asserts on the generated YAML, not just the internal attribute.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    strategy = _ar_strategy(ep=2, tp=2)
    engine_kwargs = {"pipeline_name": QWEN3_OMNI_PIPELINE, "pipeline_mode": "thinker_only"}
    strategy.preprocess_engine_kwargs(engine_kwargs)
    assert strategy._expert_parallel_stage_ids == frozenset({0})
    with open(engine_kwargs["deploy-config"]) as handle:
        deploy = yaml.safe_load(handle)
    assert deploy["stages"][0]["enable_expert_parallel"] is True

    accepted = _ar_strategy(ep=1, tp=2)
    engine_kwargs = {"pipeline_name": QWEN3_OMNI_PIPELINE, "pipeline_mode": "thinker_only"}
    accepted.preprocess_engine_kwargs(engine_kwargs)
    assert accepted._expert_parallel_stage_ids == frozenset()
    with open(engine_kwargs["deploy-config"]) as handle:
        deploy = yaml.safe_load(handle)
    assert all("enable_expert_parallel" not in stage for stage in deploy["stages"])


def test_flat_enable_expert_parallel_is_not_reinjected_from_engine_kwargs():
    """verl's launch_server overwrites the engine arg from expert_parallel_size.

    A user-provided flat ``engine_kwargs.vllm_omni.enable_expert_parallel`` must
    not survive ``prepare_engine_args``: the per-stage deploy config is the only
    supported spelling.
    """
    strategy = _ar_strategy(ep=2, tp=2)
    strategy._rollout_output_modalities = None
    strategy.server.config.engine_kwargs["vllm_omni"]["enable_expert_parallel"] = True
    engine_args = {"enable_expert_parallel": True, "model": "qwen3-omni"}
    strategy.prepare_engine_args(engine_args, types.SimpleNamespace())
    assert "enable_expert_parallel" not in engine_args


def test_pipeline_without_an_ep_opt_in_is_refused():
    strategy = _ar_strategy(ep=2, tp=2, pipeline_name="no_such_pipeline")
    with pytest.raises(ValueError, match="not supported by the selected rollout pipeline"):
        strategy.validate_configs()


def test_prepare_engine_args_drops_the_global_ep_flag():
    """A global flag would hit every stage that owns the field, non-MoE ones included.

    The generated per-stage deploy config is the single source of truth, so the
    global engine arg must not reach ``AsyncOmni``.
    """
    strategy = _ar_strategy(ep=2, tp=2)
    strategy._rollout_output_modalities = None
    engine_args = {"enable_expert_parallel": True, "model": "qwen3-omni"}
    strategy.prepare_engine_args(engine_args, types.SimpleNamespace())
    assert "enable_expert_parallel" not in engine_args


def test_refusal_names_the_supported_pipelines():
    strategy = _ar_strategy(ep=2, tp=2, pipeline_name="no_such_pipeline")
    with pytest.raises(ValueError, match=QWEN3_OMNI_PIPELINE):
        strategy.validate_configs()


def test_pipeline_that_opts_out_of_ep_is_refused(monkeypatch):
    """A registered pipeline whose hook stays empty must not silently accept EP."""

    class DenseAdapter(OmniRolloutPipelineBase):
        pass

    monkeypatch.setattr(OmniRolloutPipelineBase, "get_class", classmethod(lambda cls, model_type: DenseAdapter))
    strategy = _ar_strategy(ep=2, tp=2)
    with pytest.raises(ValueError, match="not supported by the selected rollout pipeline"):
        strategy.validate_configs()


# ---------------------------------------------------------------------------
# Diffusion rollout: dead knob, refused
# ---------------------------------------------------------------------------


def test_diffusion_ep_is_refused():
    strategy = _diffusion_strategy(ep=2)
    with pytest.raises(ValueError, match="not wired for the diffusion rollout"):
        strategy.validate_configs()


def test_diffusion_parallel_config_escape_hatch_is_refused():
    """engine_kwargs.vllm_omni.parallel_config is copied verbatim; refuse it too."""
    strategy = _diffusion_strategy(ep=1, parallel_config={"enable_expert_parallel": True})
    with pytest.raises(ValueError, match="not wired for the diffusion rollout"):
        strategy.validate_configs()


def test_diffusion_flat_enable_expert_parallel_is_refused():
    """The flat spelling reaches the diffusion engine unvalidated (live escape hatch).

    ``_diffusion_ingress_allowed_fields()`` keeps ``enable_expert_parallel``, so a
    user-set flat flag would configure EP without passing through any size check.
    """
    strategy = _diffusion_strategy(ep=1, flat_flag=True)
    with pytest.raises(ValueError, match="not wired for the diffusion rollout"):
        strategy.validate_configs()


def test_diffusion_ep_one_is_admitted():
    _diffusion_strategy(ep=1).validate_configs()
    _diffusion_strategy(ep=1, parallel_config={"enable_expert_parallel": False}).validate_configs()
