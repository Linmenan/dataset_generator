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
    def load(self, file_name: str) -> Tuple[bool, Dict[str, Dict[str, List]]]:
        """
        读取 `save_path` 目录下指定 Excel 文件。

        Parameters
        ----------
        file_name : str
            文件名(可以带或不带 .xlsx 后缀)

        Returns
        -------
        success : bool
            是否读取成功
        agents_dict : Dict[str, Dict[str, List]]
            {agent_id: {column_name: values_list}}
        """
        if not file_name.endswith(".xlsx"):
            file_name += ".xlsx"
        file_path = os.path.join(self.save_path, file_name)

        if not os.path.isfile(file_path):
            logging.error(f"文件不存在: {file_path}")
            return False, {}

        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
            agents_dict: Dict[str, Dict[str, List]] = {
                str(agent_id): {col: df[col].tolist() for col in df.columns}
                for agent_id, df in sheets.items()
            }
            return True, agents_dict
        except Exception as e:
            logging.error(f"读取 Excel 失败: {e}")
            return False, {}