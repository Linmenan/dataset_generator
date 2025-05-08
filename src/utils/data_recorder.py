import os
import pandas as pd
from datetime import datetime
import logging

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
        