#
# 樋の左右から取得した２点群の合成
# （Open3D 0.12.0 使用: pip install open3d）
#
# 2021-01-12 hshimada@ice.ous.ac.jp
import concurrent
from datetime import datetime
import open3d as o3d
import numpy as np
import copy
import sys
import asyncio

###########################################################################
###########################################################################
###########################################################################
# メイン
if __name__ == "__main__":
    print("Process start time: ", datetime.now().strftime("%H:%M:%S"))
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    target_data = "./ShellTOICUbe.xyz"
