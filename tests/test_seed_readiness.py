"""3seed学習(Gate A)前の追加レビュー指摘に対する回帰テスト (2026-10-02)。

- 方策分布: グローバルなモンキーパッチではなく factory で注入されていること、
  deterministic 推論が tanh(3*softsign(loc)) になること
- EpisodeInfoResetWrapper: done になったスロットの info 全キー・obs・pipeline_state が
  同じ fresh reset の値に揃うこと
- env の time_out: max_episode_steps に従い、転倒と同時の時間切れを含まないこと
- 足裏滑り速度: data.cvel を body 原点の速度へ正しく変換していること
- PBRS 終端: 終端stepの報酬が fall_penalty - Φ(s_prev) になること
- run_manifest 用の補助関数 (Braxのステップ算術, git provenance)
- 評価の擬似反復検出 (終端状態の指紋, Gate A 判定の警告)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("brax")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig


# ---------------------------------------------------------------------------
# 方策分布
# ---------------------------------------------------------------------------

def test_policy_distribution_is_injected_without_global_patch():
    from brax.training import distribution as brax_distribution
    from robot.policy_network import BoundedNormalTanhDistribution, make_policy_network_factory

    # import しただけで brax 側のクラスが書き換わっていないこと
    assert (brax_distribution.NormalTanhDistribution.create_dist.__qualname__
            == "NormalTanhDistribution.create_dist")
    network = make_policy_network_factory(observation_size=4, action_size=20)
    assert isinstance(network.parametric_action_distribution, BoundedNormalTanhDistribution)


def test_deterministic_inference_uses_bounded_mean():
    from brax.training.agents.ppo import networks as ppo_networks
    from robot.policy_network import POLICY_MEAN_CLIP_SCALE, make_policy_network_factory

    network = make_policy_network_factory(observation_size=4, action_size=20)
    policy_params = network.policy_network.init(jax.random.PRNGKey(0))
    obs = jp.array([[0.3, -1.2, 2.0, 0.7]])
    logits = network.policy_network.apply(None, policy_params, obs)
    raw_loc = jp.split(logits, 2, axis=-1)[0]

    make_policy = ppo_networks.make_inference_fn(network)
    policy = make_policy((None, policy_params), deterministic=True)
    action, _ = policy(obs, jax.random.PRNGKey(1))

    expected = jp.tanh(POLICY_MEAN_CLIP_SCALE * raw_loc / (1.0 + jp.abs(raw_loc)))
    np.testing.assert_allclose(np.asarray(action), np.asarray(expected), atol=1e-6)
    # 素の tanh(loc) とは異なること(テストが差を検出できる入力であることの確認)
    assert float(jp.max(jp.abs(action - jp.tanh(raw_loc)))) > 1e-3


# ---------------------------------------------------------------------------
# EpisodeInfoResetWrapper
# ---------------------------------------------------------------------------

def test_episode_info_reset_restores_all_info_obs_and_pipeline_state():
    from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper
    from envs.mjx_env import SenpuuMaruMJXEnv
    from envs.training_wrapper import EpisodeInfoResetWrapper

    env = SenpuuMaruMJXEnv()
    num_envs = 3
    wrapped = EpisodeInfoResetWrapper(
        AutoResetWrapper(EpisodeWrapper(VmapWrapper(env), episode_length=2, action_repeat=1))
    )
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(7), num_envs))
    action = jp.zeros((num_envs, env.action_size))
    step_fn = jax.jit(wrapped.step)
    state = step_fn(state, action)

    # 次stepで slot 0 だけ time limit に到達させる
    state.info["steps"] = jp.array([1.0, 0.0, 0.0])
    # wrapper 内部と同じ乱数で fresh reset を作る
    fresh_keys = jax.vmap(jax.random.split)(state.info["rng_key"])[:, 0]
    fresh = jax.jit(jax.vmap(env.reset))(fresh_keys)
    state = step_fn(state, action)

    assert np.asarray(state.done).tolist() == [1.0, 0.0, 0.0]
    for key, fresh_value in fresh.info.items():
        if key in EpisodeInfoResetWrapper.PRESERVED_KEYS:
            continue
        got = np.asarray(state.info[key])[0].astype(np.float64)
        want = np.asarray(fresh_value)[0].astype(np.float64)
        np.testing.assert_allclose(got, want, atol=1e-5, err_msg=f"info[{key!r}] was not reset")

    for key in fresh.obs:
        np.testing.assert_allclose(np.asarray(state.obs[key])[0], np.asarray(fresh.obs[key])[0], atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(state.pipeline_state.qpos)[0], np.asarray(fresh.pipeline_state.qpos)[0], atol=1e-6
    )
    # obs 末尾のサーボ温度・電圧が info の値と一致すること(旧実装は最初のepisodeの値だった)
    obs0 = np.asarray(state.obs["state"])[0]
    np.testing.assert_allclose(obs0[-21:-1], np.asarray(state.info["servo_temp"])[0], atol=1e-5)
    np.testing.assert_allclose(obs0[-1], np.asarray(state.info["supply_volt"])[0], atol=1e-5)
    # done でないスロットは継続していること
    assert np.all(np.asarray(state.info["step"])[1:] == 2)


# ---------------------------------------------------------------------------
# time_out / truncated
# ---------------------------------------------------------------------------

def test_time_out_follows_max_episode_steps():
    from envs.mjx_env import SenpuuMaruMJXEnv

    env = SenpuuMaruMJXEnv(max_episode_steps=3)
    reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    action = jp.zeros(env.action_size)
    flags = []
    for _ in range(3):
        state = step_fn(state, action)
        flags.append((bool(state.info["terminated"]), bool(state.info["truncated"]),
                      float(state.info["time_out"])))
    assert flags[0] == (False, False, 0.0)
    assert flags[1] == (False, False, 0.0)
    assert flags[2] == (False, True, 1.0)


def test_time_out_excludes_terminated_transition():
    from envs.mjx_env import SenpuuMaruMJXEnv

    env = SenpuuMaruMJXEnv(max_episode_steps=3)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    # 時間切れ直前のstepで胴体を90度ロールさせ、同じstepで転倒終了させる
    half = np.sqrt(0.5)
    qpos = state.pipeline_state.qpos.at[3:7].set(jp.array([half, half, 0.0, 0.0]))
    info = dict(state.info)
    info["step"] = jp.array(2)
    state = state.replace(pipeline_state=state.pipeline_state.replace(qpos=qpos), info=info)
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    assert bool(state.info["terminated"]) and bool(state.info["truncated"])
    assert float(state.info["time_out"]) == 0.0


# ---------------------------------------------------------------------------
# 足裏速度 / PBRS終端
# ---------------------------------------------------------------------------

def test_foot_linear_velocity_matches_mj_object_velocity():
    from envs.mjx_rewards import MJXRewardSystem

    m = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    d = mujoco.MjData(m)
    foot_ids = [i for i in range(m.nbody)
                if 'hidariashiura' in m.body(i).name or 'migiashiura' in m.body(i).name]
    assert len(foot_ids) >= 2
    d.qpos[2] = RobotConfig.INITIAL_HEIGHT
    d.qpos[3] = 1.0
    d.qvel[:] = np.random.default_rng(0).normal(size=m.nv) * 0.5
    mujoco.mj_forward(m, d)

    for foot in foot_ids:
        expected = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, foot, expected, 0)
        got = MJXRewardSystem.body_linear_velocity(
            jp.asarray(d.cvel), jp.asarray(d.xpos), jp.asarray(d.subtree_com),
            foot, int(m.body_rootid[foot]),
        )
        np.testing.assert_allclose(np.asarray(got), expected[3:6], atol=1e-5)
        # 旧実装(cvel[:,3:6]をそのまま使用)ではこの一致が成り立たない
        assert np.max(np.abs(d.cvel[foot, 3:6] - expected[3:6])) > 1e-3


def _dummy_reward_system():
    from envs.mjx_rewards import MJXRewardSystem

    class DummyModel:
        nq = 7
        nu = 6
        nsensordata = 8
        actuator_trnid = [[i, 0] for i in range(6)]
        jnt_qposadr = list(range(7, 13))
        jnt_dofadr = list(range(6, 12))

    return MJXRewardSystem(DummyModel(), RobotConfig.REWARD_WEIGHTS, left_foot_id=0, right_foot_id=1)


class _DummyData:
    def __init__(self, quat):
        self.qpos = jp.concatenate([jp.array([0.0, 0.0, 0.28]), jp.asarray(quat), jp.zeros(6)])
        self.qvel = jp.zeros(12)
        self.actuator_force = jp.zeros(6)
        self.qacc = jp.zeros(12)
        self.sensordata = jp.full(8, 1.5)
        self.xpos = jp.array([[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])
        self.subtree_com = jp.array([[0.0, 0.0, 0.28]])
        self.cvel = jp.zeros((2, 6))


def _compute(reward_system, data, last_potential):
    return reward_system.compute(
        data, action=jp.zeros(6), last_action=jp.zeros(6),
        cbf_penalty=jp.array(0.0), last_potential=jp.array(last_potential),
        step=jp.array(10),
        servo_temp=jp.full(6, 40.0), supply_volt=11.1,
        training_progress=jp.array(1.0),
    )


def test_terminal_reward_includes_pbrs_terminal_term():
    reward_system = _dummy_reward_system()
    half = float(np.sqrt(0.5))
    fallen = _DummyData([half, half, 0.0, 0.0])  # roll 90deg > TERMINATION_ROLL
    reward, done, _, _ = _compute(reward_system, fallen, last_potential=5.0)
    assert bool(done)
    expected = RobotConfig.REWARD_WEIGHTS['fall_penalty'] - 5.0
    assert float(reward) == pytest.approx(expected)

    upright = _DummyData([1.0, 0.0, 0.0, 0.0])
    _, done_upright, _, _ = _compute(reward_system, upright, last_potential=5.0)
    assert not bool(done_upright)


# ---------------------------------------------------------------------------
# run_manifest 補助関数
# ---------------------------------------------------------------------------

def test_ppo_step_arithmetic_matches_brax_formula():
    from train.train_mjx import _ppo_step_arithmetic

    arith = _ppo_step_arithmetic(
        steps=10_000_000, num_envs=256, batch_size=256, num_minibatches=16,
        unroll_length=10, num_evals=20,
    )
    assert arith["env_step_per_training_step"] == 256 * 10 * 16
    assert arith["unrolls_per_env_per_training_step"] == 16
    assert arith["num_training_steps_per_epoch"] == 13  # ceil(10M / (19 * 40960))
    assert arith["num_policy_iterations"] == 19 * 13
    assert arith["expected_total_env_steps"] == 19 * 13 * 40960


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_provenance_only_counts_real_changes_to_tracked_files(tmp_path):
    """WSL から Windows の作業ツリーを見たときの改行コード・実行権限だけの差分や、
    未追跡ファイル(評価レポート・メモ)では学習を止めない。中身の変更だけを dirty とする。"""
    from train.train_mjx import _git_provenance

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "b.py").write_bytes(b"z = 3\n")
    (tmp_path / "設定.md").write_text("メモ\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "init")

    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "out.json").write_text("{}")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "gate_a_diagnosis.md").write_text("report")
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")   # 改行コードだけ
    (tmp_path / "b.py").chmod(0o755)                          # 実行権限だけ
    prov = _git_provenance(tmp_path)
    assert prov["available"] and not prov["dirty"], prov
    assert len(prov["commit"]) == 40
    assert prov["untracked_files"] == ["docs/gate_a_diagnosis.md"]

    (tmp_path / "設定.md").write_text("変更\n", encoding="utf-8")
    (tmp_path / "a.py").write_bytes(b"x = 2\r\ny = 2\r\n")
    prov = _git_provenance(tmp_path)
    assert prov["dirty"]
    assert sorted(prov["dirty_files"]) == sorted(["a.py", "設定.md"])


# ---------------------------------------------------------------------------
# 評価の擬似反復検出
# ---------------------------------------------------------------------------

def test_state_digest_detects_identical_states():
    from scratch.phase0_eval_diagnostics import state_digest

    qpos = np.array([0.0, 0.1, 0.2])
    qvel = np.array([-0.0, 0.0])
    assert state_digest(qpos, qvel) == state_digest(qpos.copy(), np.zeros(2))
    assert state_digest(qpos, qvel) != state_digest(qpos + 1e-3, qvel)


def _write_report(path, n, n_unique):
    report = {
        "checkpoint": str(path),
        "conditions": {
            "deterministic__randomized_dr": {
                "n_episodes": n, "success_rate": 1.0, "n_unique_final_states": n_unique,
            }
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_gate_a_qualification_flags_pseudo_replication(tmp_path):
    script = ROOT / "scratch" / "gate_a_qualification.py"
    ok = tmp_path / "ok.json"
    dup = tmp_path / "dup.json"
    _write_report(ok, n=200, n_unique=200)
    _write_report(dup, n=200, n_unique=1)

    clean = subprocess.run(
        [sys.executable, str(script), "--reports", str(ok), "--threshold", "0.95"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "判定: PASS" in clean and "参考判定" not in clean

    flagged = subprocess.run(
        [sys.executable, str(script), "--reports", str(ok), str(dup), "--threshold", "0.95"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "終端状態の重複あり" in flagged and "参考判定" in flagged
