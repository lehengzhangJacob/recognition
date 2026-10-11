# Y7000P 前置摄像头：商品确认与 SQLite 入库

本地原型使用 RF-DETR 三分类 ONNX，推理只使用 CPU。前置摄像头在 OpenCV 中通常是 `--camera 0`。它把 **笔、笔记本、笔袋**分别记成 `pen`、`notebook`、`pencil_pouch`。

## 在 Windows 上准备

1. 安装 Python 3.10、3.11 或 3.12，下载或克隆本仓库，然后在仓库根目录打开 PowerShell。
2. 安装 CPU 依赖：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r models\stationery-v1\requirements-cpu.txt
```

3. 模型 ONNX 因 GitHub 文件大小限制分成两段。先还原并验证哈希：

```powershell
.\.venv\Scripts\python.exe models\stationery-v1\scripts\assemble_rf_detr_artifacts.py
```

4. 运行摄像头确认程序：

```powershell
.\.venv\Scripts\python.exe models\stationery-v1\scripts\vending_confirm_camera.py --camera 0
```

如果打开的不是前置摄像头，改为 `--camera 1`。默认请求 1280×720；窗口会显示摄像头实际输出的分辨率。

## 使用

摄像头应固定。让画面中央的灰色框保持空白，**按空格模拟一次售货机出货信号**，然后把物品放进灰框。程序检查模型分数、三类分数差距、目标框是否在灰框内、目标框相对按空格时的画面是否确实改变，以及最近 7 帧中是否有至少 5 帧判为同一类。确认成功后写入 SQLite 一条记录；这次出货不能重复入库。移走物品，等待状态回到 `IDLE`，再按空格确认下一件。按 `Q` 或 Esc 退出。

注意：按空格时灰框要是空的，因为这一帧是变化检测的基准。如果物品已经在画面里，程序会等待新的变化，不会马上入库。`--roi "0.15,0.15,0.85,0.85"` 可修改识别区域。摄像头面对真实出货口时，应把这个区域收紧到出货通道。

## 查看数据库

默认数据库位于 `models/stationery-v1/runtime/confirmations.sqlite3`。查看最近 30 条：

```powershell
.\.venv\Scripts\python.exe models\stationery-v1\scripts\vending_confirm_camera.py --list
```

`confirmations` 表记录 `dispense_id`、UTC 确认时间、商品类别、分数、分数差距、目标框、模型版本和摄像头编号。`dispense_id` 是主键，重复提交同一编号不会多写一条。将来接售货机控制器时，让控制器在发出货动作时调用 `ConfirmController.arm(dispense_id)`，把实际出货单号传入。

## 调整与限制

`--pen-conf`、`--notebook-conf`、`--pouch-conf` 调整各类别最低模型分数；`--min-margin` 调整最高类与第二类的分数差；`--min-change` 调整目标框内相对空画面的变化比例。默认值仅供前置摄像头试用，正式售货机需要采集真实出货视频调参，并测试空画面、手遮挡、连续出货和不同款式的笔袋。笔袋目前的验证图来自同一只笔袋，不能仅凭现有准确率直接当作现场保证。
