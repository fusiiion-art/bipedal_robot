# 外乱耐性直立姿勢維持 — タスクリスト

## Phase 1: 基本直立の安定化

- [ ] `config.py`: ACTION_SCALE を ±90° → ±30° に縮小
- [ ] `config.py`: REWARD_WEIGHTS の再調整 (alive増、fall_penalty緩和、ペナルティ縮小)
- [ ] `config.py`: MAX_EPISODE_STEPS を 1000 → 500 に短縮
- [ ] `config.py`: RANDOM_PUSH_MAX_FORCE を 0.0 に (外乱無効化)
- [ ] `config.py`: DR範囲の縮小 (RANDOM_MASS_SCALE, RANDOM_FRICTION, RANDOM_COM_OFFSET)
- [ ] `mjx_env.py`: global_step を training_progress ベースに修正 (CRITICAL-1 バグ修正)
- [ ] `mjx_rewards.py`: ペナルティスケールダウン + alive/upright 集中報酬
- [ ] `train_mjx.py`: reward_scaling 調整、パラメータ確認
- [ ] Phase 1 の学習テスト実行・結果確認

## Phase 2: 外乱耐性の段階的導入
- [ ] `config.py`: CURRICULUM_SCHEDULE を training_progress ベースに切り替え
- [ ] `mjx_env.py`: _get_curriculum_scale を training_progress ベースに変更
- [ ] `config.py`: 外乱を段階的に導入 (0% → 100%)
- [ ] Phase 2 の学習テスト実行・結果確認

## Phase 3: 回復行動の強化
- [ ] `mjx_rewards.py`: 回復報酬系のウェイト引き上げ
- [ ] `config.py`: RECOVERY_BONUS_STABILITY_THRESHOLD を 0.7 → 0.4 に緩和
- [ ] Phase 3 の学習テスト実行・結果確認
