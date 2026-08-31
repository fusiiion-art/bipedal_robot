#!/bin/bash
# =============================================================
# real/setup/install_rt.sh — RT-Preemptカーネルのセットアップ手順
# =============================================================
# 
# 【目的】
# 汎用Linuxカーネルのスケジューリングジッター (数ms〜数十ms) を排除し、
# 100Hz (10ms) の制御ループを安定稼働させるためのリアルタイム化。
#
# 【RPi5での実行手順】
#   chmod +x install_rt.sh
#   sudo ./install_rt.sh
# =============================================================

set -euo pipefail

echo "============================================"
echo " RPi5 RT-Preempt セットアップ"
echo "============================================"

# --- 1. 必要パッケージのインストール ---
echo "[1/5] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    git bc bison flex libssl-dev make \
    libncurses5-dev libelf-dev \
    python3-pip python3-venv

# --- 2. PREEMPT_RT パッチ済みカーネルのインストール ---
# Raspberry Pi OS (64-bit) の場合、公式の Raspberry Pi Linux ソースに RT パッチを適用してビルドします。
# もしくは、事前にビルドされた RT パッケージを適用してください。
echo "[2/5] Setting up PREEMPT_RT kernel guide for Raspberry Pi OS..."
echo "  Raspberry Pi 5 (kernel_2712) requires a customcompiled RT-kernel."
echo "  See: https://www.raspberrypi.com/documentation/computers/linux_kernel.html"
echo "  Applying generic RT guidance..."
sudo apt-get install -y rt-tests || {
    echo "[Info] rt-tests package skipped or not available."
}

# --- 3. ブートパラメータの設定 (CPUコア分離) ---
echo "[3/5] Configuring CPU isolation..."
BOOT_CMDLINE="/boot/firmware/cmdline.txt"
if [ -f "$BOOT_CMDLINE" ]; then
    if ! grep -q "isolcpus=3" "$BOOT_CMDLINE"; then
        sudo sed -i 's/$/ isolcpus=3 nohz_full=3 rcu_nocbs=3/' "$BOOT_CMDLINE"
        echo "  Added: isolcpus=3 nohz_full=3 rcu_nocbs=3"
    else
        echo "  Already configured."
    fi
fi

# --- 4. Python仮想環境とランタイム依存関係 ---
echo "[4/5] Setting up Python virtual environment..."
VENV_DIR="/opt/bipedal_runtime/venv"
sudo mkdir -p /opt/bipedal_runtime
sudo python3 -m venv "$VENV_DIR"
sudo "$VENV_DIR/bin/pip" install --upgrade pip
sudo "$VENV_DIR/bin/pip" install \
    onnxruntime \
    numpy \
    pyserial \
    spidev \
    pyyaml

# --- 5. systemd サービスのインストール ---
echo "[5/5] Installing systemd services..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# メインループ サービス
sudo tee /etc/systemd/system/bipedal_main.service > /dev/null << 'EOF'
[Unit]
Description=Bipedal Robot Main Control Loop (100Hz)
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/chrt -f 99 /usr/bin/taskset -c 3 \
    /opt/bipedal_runtime/venv/bin/python3 -m real.real_env
WorkingDirectory=/opt/bipedal_runtime/src
Restart=on-failure
RestartSec=2
ExecStopPost=/usr/bin/find /dev/shm -name "robot_*" -delete

# リソース制限
LimitRTPRIO=99
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# RMA適応器 サービス
sudo tee /etc/systemd/system/bipedal_rma.service > /dev/null << 'EOF'
[Unit]
Description=Bipedal Robot RMA Adaptation Module (20Hz)
After=bipedal_main.service
Requires=bipedal_main.service

[Service]
Type=simple
User=root
ExecStart=/usr/bin/chrt -f 90 /usr/bin/taskset -c 2 \
    /opt/bipedal_runtime/venv/bin/python3 -m real.rma_worker
WorkingDirectory=/opt/bipedal_runtime/src
Restart=on-failure
RestartSec=3
ExecStopPost=/usr/bin/find /dev/shm -name "robot_*" -delete

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo ""
echo "============================================"
echo " セットアップ完了!"
echo ""
echo " 使い方:"
echo "   sudo systemctl start bipedal_main"
echo "   sudo systemctl start bipedal_rma"
echo ""
echo " 自動起動を有効化:"
echo "   sudo systemctl enable bipedal_main"
echo "   sudo systemctl enable bipedal_rma"
echo ""
echo " ログ確認:"
echo "   journalctl -u bipedal_main -f"
echo ""
echo " ⚠️  再起動が必要です (CPUアイソレーション適用のため)"
echo "   sudo reboot"
echo "============================================"
