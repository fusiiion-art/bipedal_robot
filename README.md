# 二足歩行ロボット『旋風丸』ドキュメントインデックス（決定・全不整合解決版）

二足歩行ロボット『旋風丸 (Senpumaru_GIY_Type)』の設計、開発計画、ハードウェア構成、強化学習、実機デプロイに関する統合ドキュメント一覧です。

---

## 📁 フォルダ構成とドキュメントマップ

### 1. 01_plans/ (開発計画・設計仕様)
- 📄 [旋風丸_開発計画書_完全版.md](file:///c:/bipedal_robot/docs/01_plans/旋風丸_開発計画書_完全版.md) **【マスター開発計画書 (SSOT)】**  
  全世代・Sim-to-Real統合・理論・電気・制御を網羅したマスタープラン v1.1.2。
- 📄 [旋風丸_V1_Biped_統合開発計画書_決定版.md](file:///c:/bipedal_robot/docs/01_plans/旋風丸_V1_Biped_統合開発計画書_決定版.md) **【V1実機仕様（完全決定版）】**  
  Teensy 4.1 脊髄実装 (TeensySpineIO)、3S LiPo 11.1V、20自由度（頭部なし）、FSRオンチップADC二値判定を確定したV1仕様書。
- 📄 [旋風丸_V1_設計レビュー統合.md](file:///c:/bipedal_robot/docs/01_plans/旋風丸_V1_設計レビュー統合.md) **【評価レビューアーカイブ】**  
  外部AIおよび査定結果の集約。
- 📄 [PROJECT_GOAL.md](file:///c:/bipedal_robot/docs/01_plans/PROJECT_GOAL.md)  
  核心的ビジョンとプロジェクト目標。

---

### 2. 02_hardware/ (ハードウェア・部品構成)
- 📄 [HARDWARE_SPECS_AND_BOM.md](file:///c:/bipedal_robot/docs/02_hardware/HARDWARE_SPECS_AND_BOM.md) **【統合ハードウェア仕様 ＆ 改訂版BOM】**  
  3S LiPo (11.1V)、5V/5A 降圧UBEC、40A ヒューズ、AWG12 配線、AE-LLCNV-LVCH16T245 レベル変換、FSR×8、完全BOM表。

---

### 3. 03_manuals/ (システム運用マニュアル)
- 📄 [学習側説明書.md](file:///c:/bipedal_robot/docs/03_manuals/学習側説明書.md)  
  MuJoCo MJX ＋ Brax PPO 学習ガイド。
- 📄 [実機側説明書.md](file:///c:/bipedal_robot/docs/03_manuals/実機側説明書.md)  
  Phase 0 〜 Phase 3（Phase 0.5 SILダミー検証含む）実機デプロイマニュアル。

---

### 4. 04_rl_rewards/ (強化学習・報酬関数設計)
- 📄 [報酬関数_詳細仕様書.md](file:///c:/bipedal_robot/docs/04_rl_rewards/報酬関数_詳細仕様書.md)  
  位相ゲート適用 PBRS、2段階 Capture Point 報酬、ZMP 符号付き距離マージン、対数バリア仕様書。
- 📄 [REWARD_IMPROVEMENTS.md](file:///c:/bipedal_reward/docs/04_rl_rewards/REWARD_IMPROVEMENTS.md)  
  5重大致命的バグの解消および検証レポート。
