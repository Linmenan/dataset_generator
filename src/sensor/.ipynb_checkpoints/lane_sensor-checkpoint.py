from rtree import index
from typing import Tuple, List
import os
import shutil
import numpy as np

from src.models.map_elements import Lane
from lib.geometry_lib.point.point import Point2D
from src.parsers.parsers import MapParser

class LaneSensor:
    def __init__(
            self, 
            map_parser: MapParser, 
            index_file:str = 'road_network_idx'
            ):
        self.map_parser = map_parser
        self.index_file = index_file
        self.clean_index_files()
        self.query_engine = QueryEngine(self.index_file)
        self.build_engine()

    # @property
    def build_engine(self):
        lanes=[]
        for road in self.map_parser.roads.values():
            for lane in road.lanes:
                lanes.append(lane)
        self.query_engine.build(lanes)
    # 按照地图坐标系位置与范围，感知进入范围的所有车道对象
    def radius_lanes_query(self, center:Point2D, radius:float)->List[Lane]:
        """返回区域所有的车道Lane"""
        result = self.query_engine.radius_query(center, radius)
        ref_pts=[]
        for lane in result[0]:
            ref_pts = [Point2D(x, y) for (x, y), (_, _), (_, _) in lane.sampled_points]
        return result[0]
    # 按照地图坐标系位置与范围，感知进入范围的所有车道路点
    def radius_lanes_points_query(self, center:Point2D, radius:float)->List[Tuple[List[Point2D],List[Point2D],List[Point2D]]]:
        result,distance = self.query_engine.radius_query(center, radius)
        refs=[]
        for lanes,diss in zip(result,distance):
            lane_pts=[(Point2D(x, y),Point2D(xl, yl),Point2D(xr, yr)) for ((x,y),(xl,yl),(xr,yr)),dis in zip(lanes.sampled_points,diss) if dis<= radius]
            refs.append(lane_pts)
        return refs
    # @property
    def clean_index_files(self):
        for ext in ['.dat', '.idx']:
            file_path = self.index_file + ext
        try:
            if os.path.exists(file_path):
                # print("删除:",file_path)
                os.unlink(file_path)  # 比 os.remove 更安全
        except Exception as e:
            print(f"删除 {file_path} 失败: {str(e)}")

class SpatialIndex:
    def __init__(
            self,
            index_file = 'road_network_idx'):
        # 配置索引参数
        self.properties = index.Property()
        self.properties.overwrite = True  # 🔥 关键修复参数
        self.properties.storage = index.RT_Disk
        self.properties.dimension = 2    # 二维坐标
        self.properties.buffering_capacity = 64  # 磁盘缓存块大小
        self.properties.leaf_capacity = 50       # 叶节点容量
        self.properties.index_capacity = 100     # 内部节点容量

        # 创建持久化索引
        self.idx = index.Index(index_file, 
                            interleaved=False,  # 坐标顺序(x, y)
                            properties=self.properties)
        # 内存数据存储
        self.lane_map = {}

    def build_index(self, lanes:List[Lane]):
        """批量插入道路数据"""
        # 生成索引条目 (id, (min_x, min_y, max_x, max_y), None)
        print("加载车道数量：",len(lanes))
        for lane in lanes: 
            self.idx.insert(int(lane.unicode), lane.mbr, None)
            # print("id：",lane.unicode)
        
        # 存储完整数据
            self.lane_map.update({int(lane.unicode): lane for lane in lanes})


class QueryEngine:
    def __init__(
            self,
            index_file = 'road_network_idx'):
        self.index = SpatialIndex(index_file)
        self.cache = {}  # 查询结果缓存

    def build(self,lanes):
        self.index.build_index(lanes)

    def range_query(self, bbox, use_cache=True)->List[Lane]:
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

    def radius_query(self, center:Point2D, radius:float)->Tuple[List[Lane],List[np.ndarray]]:
        """
        圆形范围查询（转换为矩形过滤+精确计算）
        :param center: (x, y) 中心点坐标
        :param radius: 查询半径（单位与坐标一致）
        """
        # 第一步：MBR快速过滤
        search_bbox = (
            center.x - radius, 
            center.x + radius,
            center.y - radius,
            center.y + radius
        )
        candidates = self.range_query(search_bbox, use_cache=False)
        
        # 第二步：精确计算
        results = []
        results_distance = []
        for lane in candidates:
            distances = np.sqrt(
                (lane.coords[:,0] - center.x)**2 + 
                (lane.coords[:,1] - center.y)**2
            )
            # print("最小距离：",np.min(distances))

            if np.any(distances <= radius):
                # print('成功')
                results.append(lane)
                results_distance.append(distances)
        
        return (results,results_distance)
    
