from ..parsers.parsers import MapParser
from ..models.map_elements import *
import numpy as np
from rtree import index
import os
import shutil

class SpatialIndex:
  def __init__(self):
    # 配置索引参数
    self.properties = index.Property()
    self.properties.overwrite = True  # 🔥 关键修复参数
    self.properties.storage = index.RT_Disk
    self.properties.dimension = 2    # 二维坐标
    self.properties.buffering_capacity = 64  # 磁盘缓存块大小
    self.properties.leaf_capacity = 50       # 叶节点容量
    self.properties.index_capacity = 100     # 内部节点容量
    
    # 创建持久化索引
    self.idx = index.Index('road_network_idx', 
                          interleaved=False,  # 坐标顺序(x, y)
                          properties=self.properties)
    
    # 内存数据存储
    self.lane_map = {}

  def build_index(self, lanes):
    """批量插入道路数据"""
    # 生成索引条目 (id, (min_x, min_y, max_x, max_y), None)
    print("车道数量：",len(lanes))
    for lane in lanes: 
      self.idx.insert(int(lane.lane_id), lane.mbr, None)
    
    # 存储完整数据
      self.lane_map.update({int(lane.lane_id): lane for lane in lanes})


class QueryEngine:
  def __init__(self):
    self.index = SpatialIndex()
    self.cache = {}  # 查询结果缓存

  def build(self,lanes):
    self.index.build_index(lanes)

  def range_query(self, bbox, use_cache=True):
    """
    矩形范围查询
    :param bbox: 查询范围 (min_x, min_y, max_x, max_y)
    :return: 符合条件的所有RoadSegment
    """
    cache_key = tuple(np.round(bbox, 2))
    if use_cache and cache_key in self.cache:
        return self.cache[cache_key]
    
    # 执行索引查询
    road_ids = list(self.index.idx.intersection(bbox))
    print("检索到车道数量：",len(road_ids))

    results = [self.index.lane_map[int(rid)] for rid in road_ids]
    
    # 缓存结果（LRU策略）
    self.cache[cache_key] = results
    if len(self.cache) > 1000:
        self.cache.popitem(last=False)
    
    return results

  def radius_query(self, center, radius):
    """
    圆形范围查询（转换为矩形过滤+精确计算）
    :param center: (x, y) 中心点坐标
    :param radius: 查询半径（单位与坐标一致）
    """
    # 第一步：MBR快速过滤
    search_bbox = (
        center[0] - radius, 
        center[0] + radius,
        center[1] - radius,
        center[1] + radius
    )
    candidates = self.range_query(search_bbox, use_cache=False)
    
    # 第二步：精确计算
    results = []
    for lane in candidates:
        distances = np.sqrt(
            (lane.coords[:,0] - center[0])**2 + 
            (lane.coords[:,1] - center[1])**2
        )
        print("最小距离：",np.min(distances))

        if np.any(distances <= radius):
            print('成功')
            results.append(lane)
    
    return results

class LaneSimulator:
  def __init__(
            self, 
            map_file_path='', 
            yaml_path='', 
            perception_range=0, 
            ):
        self.index_file = 'road_network_idx'
        self.perception_range = perception_range
        self.clean_index_files()
        self.query_engine = QueryEngine()
        self.ego_vehicle = None
        self.map_parser = MapParser(file_path=map_file_path,yaml_path=yaml_path)
        print("lane init")
        self.build_engine()

  def build_engine(self):
    lanes=[]
    for road in self.map_parser.roads.values():
      for lane in road.lanes:
        if int(lane.lane_id)<0:
          id = abs(int(lane.lane_id))+10
        else:
          id = int(lane.lane_id)
        lane.lane_id = road.road_id+'99999'+str(id)
        lanes.append(lane)
  
    self.query_engine.build(lanes)

  def radius_query(self, center, radius):
    return self.query_engine.radius_query(center, radius)

  def clean_index_files(self):
    """原子化删除操作"""
    print("原子化删除操作")
    for ext in ['.dat', '.idx']:
      file_path = self.index_file + ext
      try:
          if os.path.exists(file_path):
              print("删除:",file_path)
              os.unlink(file_path)  # 比 os.remove 更安全
      except Exception as e:
          print(f"删除 {file_path} 失败: {str(e)}")
    


