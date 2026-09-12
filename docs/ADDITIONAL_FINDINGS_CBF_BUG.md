# 追加コード検査レポート — 「他にないか」への回答

**実施日**: 2026-09-11  
**内容**: 前回レポート後の深掘り検査で新たに発見・修正した事項

---

## 🚨 【最重要】実際のバグを1件発見・修正しました

### `safety/cbf.py` の `compute_cbf_penalty()` 呼び出しミスマッチ

#### 発見の経緯
`gate0_formal_eval.py` の類似問題（ゼロ行動評価）がないか、他のスクリプトを横断確認していたところ、**関数の定義と呼び出しの引数が一致しない箇所**をAST解析で発見しました。

#### バグの内容

**定義**（`safety/cbf.py` 118行目）:
```python
def compute_cbf_penalty(
    self,
    nominal_action: jp.ndarray,
    safe_action: jp.ndarray,        # ← 2番目の引数
    limit_lower: jp.ndarray = None,
    limit_upper: jp.ndarray = None
) -> jp.ndarray:
```

**旧・呼び出し**（`envs/mjx_env.py` 294行目、修正前）:
```python
cbf_penalty = self._cbf.compute_cbf_penalty(target_rad, limit_lower, limit_upper)
#                                                        ↑ここが safe_action の位置なのに
#                                                          limit_lower(関節下限)が渡っていた
```

#### 何が起きていたか

引数が1つずつズレて渡っていました：

| 引数位置 | 関数が期待するもの | 実際に渡っていたもの |
|---|---|---|
| 1番目 | `nominal_action` | `target_rad` ✅（正しい） |
| 2番目 | `safe_action`（クランプ後の安全アクション） | `limit_lower`（関節下限） ❌ |
| 3番目 | `limit_lower`（関節下限） | `limit_upper`（関節上限） ❌ |
| 4番目 | `limit_upper`（関節上限） | **渡されず** → `None` ❌ |

#### 実害

1. **`direct_penalty`** が `|target_rad - safe_action|`（クランプでどれだけ削られたか）ではなく `|target_rad - limit_lower|`（目標角と関節下限との距離）という**無意味な量**を計算していた
2. **margin-basedのsoftplusペナルティ**（コード内で「CBF-2/CBF-3 FIXED」として導入されたはずのより厳格な項）が、`limit_upper` が `None` のままだったため**常にスキップ**されていた

つまり、`safety/cbf.py` 内のコメントに書かれている「filter_action() と compute_cbf_penalty() のペナルティ基準を統一した」という修正意図が、**呼び出し側の更新漏れにより実際には反映されていませんでした**。

#### 影響範囲の切り分け

- ❌ **影響あり**: CBFペナルティによる報酬整形（RLが「クランプされないよう」学習する誘導効果）
- ✅ **影響なし**: 物理的な安全性そのもの — `filter_action()` によるハードクランプは正しく別途適用されており、実際の関節角がリミットを超えることは防がれていました

**つまり「関節が壊れる」等の直接的な危険はありませんでしたが、学習の質（CBFを避けるような滑らかな動きの獲得）に悪影響があった可能性があります。**

#### 修正内容

```python
# 修正後
safe_target_rad = self._cbf.filter_action(filtered_action, limit_lower, limit_upper)
cbf_penalty = self._cbf.compute_cbf_penalty(target_rad, safe_target_rad, limit_lower, limit_upper)
#                                                        ^^^^^^^^^^^^^^^ 正しく safe_target_rad を渡す
```

詳細な経緯をコードコメントとして残し、将来同じ間違いが再発しないようにしています。

#### 検証
- ✅ 構文チェックOK
- ✅ 単体テスト19/19 PASS（退行なし）
- ⚠️ **この修正は「報酬の中身が変わる」変更なので、改良規約 §16「変更単位」に従い、既存のcheckpoint評価とは区別して扱ってください**（もし過去にこのバグ入りコードで学習したcheckpointがあれば、そのcheckpointの評価結果とは報酬の意味が変わっています）

---

## 🔍 横断検査で確認した「バグではなかった」項目

AST解析で他に3件の「引数数不一致」候補が出ましたが、全て**同名メソッドが複数クラスに存在するための誤検知**と判明しました：

| 候補 | 実際 |
|---|---|
| `training_wrapper.py:63` の `__init__()` | `TrainingProgressWrapper` 自身の `__init__`（`CBFSafetyFilter.__init__` とは無関係） |
| `training_wrapper.py:103` の `step()` | Brax環境の `step(state, action)`（`real_env.py` とは無関係） |
| `mjx_env.py:373` の `step()` | MJXライブラリ自体の `mjx.step(model, data)`（`real_env.py` とは無関係） |

---

## 📋 その他のスクリプトのゼロ行動チェック（gate0_formal_eval.pyと同種の問題がないか）

| スクリプト | ゼロ行動使用 | Checkpoint読込 | 判定 |
|---|---|---|---|
| `gate0_mujoco_eval.py` | ✅ PD制御（default_pose保持） | 不要 | ✅ **正しい設計**（Gate 0-P = 純MuJoCo物理ベースライン、docs定義通り） |
| `gate0_standing_eval.py` | ⚠️ ゼロ行動、5stepのみ | なし | ⚠️ **要判断**（下記参照） |
| `gate0_formal_eval.py` | ✅ 修正済み（前回対応） | ✅ あり | ✅ 修正済み |
| `phase0_ppo_diagnostics.py` | - | - | policy生成ロジックなし、別種の診断スクリプト |
| `validate_policy_bounds.py` | - | - | 境界値検証専用、行動生成不要 |

### `gate0_standing_eval.py` について

このスクリプトは **5ステップ（0.05秒）だけの簡易物理チェック**で、`gate0_mujoco_eval.py`（Gate 0-P、10秒間のPD評価）と役割が重複している可能性があります。おそらく開発初期の実験的スクリプトで、後から作られた `gate0_mujoco_eval.py` と `gate0_formal_eval.py` に役割を譲った「残骸」の可能性が高いです。

**削除するかどうかはあなたの判断が必要です**（改良規約により、判断を要する削除は自動実行しません）。

---

## 🧹 未使用import（低優先度、未適用）

コード動作に影響しない軽微な項目です。適用はしていません（別カテゴリの変更のため、改良規約 §16 に従い分離）：

| ファイル | 未使用import |
|---|---|
| `robot/config.py` | `os` |
| `robot/gait_generator.py` | `Optional` |
| `envs/actuator_model.py` | `RobotConfig` |
| `envs/mjx_env.py` | `Union` |
| `real/real_env.py` | `Dict` |
| `train/export_trajectory.py` | `RobotConfig`, `jnp` |
| `train/view_trajectory.py` | `ctypes` |
| `train/visualize_rl.py` | `jp` |
| `train/train_mjx.py` | `datetime`（私の編集前から存在） |

※ `train/train_mjx.py` と `train/export_trajectory.py` の `SenpuuMaruMJXEnv` importは `# noqa: F401` 付きで**意図的**（Brax環境登録の副作用）と確認済みのため、リストから除外しています。

**希望があれば、これらをまとめて削除する別iterationとして対応できます。**

---

## ✅ 確認して「問題なし」だった項目

- `real/real_io.py` の checksum 検証: **既に [REAL-4 FIXED] として正しく厳格化済み**
- `real/real_io.py` の IMU異常値保護（BNO055UART.get_quaternion）: ノルムチェック・直前有効値保持ともに正しく実装
- `calc_checksum()` のビット演算: `~sum(buf) & 0xFF` は標準的なHiwonder方式と一致、オーバーフロー等の問題なし
- TODO/FIXME/XXX/HACKコメント: プロジェクト全体で実質0件（クリーンなコードベース）

---

## 📊 総括

| カテゴリ | 発見数 | 対応 |
|---|---|---|
| 🚨 実際のバグ（動作に影響） | **1件**（CBFペナルティ引数ミス） | ✅ **修正済み** |
| ⚠️ 要判断（削除等） | 2件（test_gui.py、gate0_standing_eval.py） | 報告のみ |
| 🧹 軽微（未使用import） | 9件 | 報告のみ、未適用 |
| ✅ 検証して問題なしと確認 | 4件 | - |

---

## 📁 更新された出力ファイル

`/mnt/user-data/outputs/improved_files/mjx_env.py` を **CBF修正込みのバージョン**に更新しました（NaN検出のdocstring改良に加えて、今回のバグ修正を含む最新版です）。

```
improved_files/
  ├─ train_mjx.py          (前回同様: NaN検出3段階 + module docstring)
  ├─ mjx_env.py             ← 🆕 CBFバグ修正を追加 (26 KB、更新版)
  ├─ mjx_rewards.py         (前回同様: reward_is_finiteフラグ + module docstring)
  └─ gate0_formal_eval.py   (前回同様: 学習済み方策対応)
```

---

## 🎯 次のステップ（更新版）

```
1. ✅ 全方位再検査完了
2. ✅ NaN/Inf即時停止機構を実装
3. ✅ CBFペナルティのバグを発見・修正 ← 🆕
4. ✅ Docstring改良を実装
5. ✅ 単体テスト19/19 PASS確認（バグ修正後も retest 済み）
6. 🔜 test_gui.py / gate0_standing_eval.py の削除要否をあなたが判断
7. 🔜 未使用importの削除（希望があれば別iterationで対応）
8. 🔜 修正ファイルをC:\bipedal_robotに適用
9. 🔜 D-6 GPU Debug run 実行（改良後のコードで）
```

---

**報告者**: Claude  
**実施日**: 2026-09-11
