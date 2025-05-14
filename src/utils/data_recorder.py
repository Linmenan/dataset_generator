import os
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple

class DataRecorder:
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.data = {}
        self.current_line = {}
        self.work_on = True
    def add_data(self, tab, name, value):
        if not self.work_on:
            return
        if tab not in self.data:
            self.data[tab] = []
            self.current_line[tab] = {}
        self.current_line[tab][name] = value

    def new_line(self):
        if not self.work_on:
            return
        for tab in self.data:
            line = self.current_line.get(tab, {})
            self.data[tab].append(line)
            self.current_line[tab] = {}

    def save(self):
        if not self.work_on:
            return
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(self.save_path, f"{now}.xlsx")
        try:
            with pd.ExcelWriter(file_path) as writer:
                for tab, lines in self.data.items():
                    df = pd.DataFrame(lines)
                    df.to_excel(writer, sheet_name=tab, index=False)
            logging.info(f"数据成功保存到 {file_path}")
        except Exception as e:
            logging.error(f"数据保存失败: {e}")
        with pd.ExcelWriter(file_path) as writer:
            for tab, lines in self.data.items():
                df = pd.DataFrame(lines)
                df.to_excel(writer, sheet_name=tab, index=False)
        
    # ------------------ 读取相关 ------------------
    def load(
        self,
        file_name: str | None = None,
        exit_cmds: Tuple[str, ...] = ("q", "quit", "exit"),
    ) -> Tuple[bool, Dict[str, Dict[str, List]]]:
        """
        交互式读取 save_path 目录下的 Excel 文件。

        Parameters
        ----------
        file_name : str | None
            - 传入合法文件名时将直接尝试读取；
            - 传入 None 或非法时，会进入交互模式。
        exit_cmds : Tuple[str, ...]
            用户在交互模式下输入任一指令即可退出并返回 (False, {})。

        Returns
        -------
        (success, agents_dict)
        """
        # 准备 Excel 列表
        excel_files = [
            f for f in os.listdir(self.save_path) if f.lower().endswith(".xlsx")
        ]
        if not excel_files:
            logging.error(f"目录 {self.save_path} 下没有找到 .xlsx 文件")
            return False, {}

        def _is_valid(name: str) -> bool:
            return name in excel_files

        # 如果传入了文件名但不合法，强制进入交互
        if file_name and not _is_valid(file_name):
            print(f"⚠️  文件 {file_name} 不存在，将进入交互选择…")
            file_name = None

        # ---------- 交互循环 ----------
        while file_name is None:
            print("\n可选数据文件：")
            for idx, fname in enumerate(excel_files, 1):
                print(f"[{idx}]  {fname}")
            choice = input("请输入序号或文件名（输入 q 退出）：").strip()

            # 退出判断
            if choice.lower() in exit_cmds:
                print("已退出数据加载。")
                return False, {}

            # 数字序号
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(excel_files):
                    file_name = excel_files[idx - 1]
                    break
                else:
                    print("❌ 序号超出范围，请重新输入。")
                    continue

            # 文件名
            if _is_valid(choice):
                file_name = choice
                break

            print("❌ 输入无效，请重新输入。")

        # ---------- 正式读取 ----------
        file_path = os.path.join(self.save_path, file_name)
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
            agents_dict: Dict[str, Dict[str, List]] = {
                str(agent_id): {col: df[col].tolist() for col in df.columns}
                for agent_id, df in sheets.items()
            }
            logging.info(f"成功读取 {file_path}")
            return True, agents_dict

        except Exception as e:
            logging.error(f"读取 Excel 失败: {e}")
            return False, {}