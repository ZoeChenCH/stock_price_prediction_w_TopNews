#!/bin/bash

# 設定時間與路徑
PYTHON_PATH="/Users/zhao-weichen/anaconda3/envs/torch/bin/python"
SCRIPT_PATH="/Users/zhao-weichen/Zoe/Strock_tracking/News_classification_UpDown/daily_news.py"
LOG_PATH="/Users/zhao-weichen/news_log.txt"

# crontab 時間設定：每天 14:00 執行
CRON_TIME="0 14 * * *"

# 組合 crontab 指令
CRON_JOB="$CRON_TIME $PYTHON_PATH $SCRIPT_PATH >> $LOG_PATH 2>&1"

# 檢查是否已經存在一樣的排程
(crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH") >/dev/null
if [ $? -eq 0 ]; then
  echo "已經存在相同的 crontab 任務，略過新增"
else
  (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
  echo "已新增 crontab 排程："
  echo "$CRON_JOB"
fi
