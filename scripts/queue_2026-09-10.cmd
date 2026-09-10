@echo off
REM Experiment queue for 10 Sep 2026. Runs sequentially and detached from any terminal:
REM   powershell -Command "Start-Process -FilePath scripts\queue_2026-09-10.cmd -WorkingDirectory . -WindowStyle Hidden"
REM Progress: runs\matrix_q*.log ; finished runs have runs\<name>\test_metrics.json.
cd /d %~dp0\..
set P=.venv\Scripts\python.exe

:wait_frozen
if not exist runs\prithvi_frozen_f1.00_s0\test_metrics.json (
  timeout /t 30 /nobreak >nul
  goto wait_frozen
)

%P% scripts\run_matrix.py --models prithvi_lora --fractions 1.0 --seeds 0 > runs\matrix_q1_lora.log 2>&1
%P% scripts\run_matrix.py --models prithvi_full --fractions 1.0 --seeds 0 > runs\matrix_q2_full.log 2>&1
%P% scripts\run_matrix.py --models prithvi_frozen prithvi_lora --fractions 0.25 0.10 0.05 --seeds 0 > runs\matrix_q3_prithvi_sweep.log 2>&1
%P% scripts\run_matrix.py --models unet_scratch unet_imagenet --fractions 1.0 0.25 0.10 0.05 --seeds 1 2 > runs\matrix_q4_unet_seeds.log 2>&1
echo QUEUE DONE > runs\queue_2026-09-10.done
