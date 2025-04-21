from ..models.agent import *
from ..parsers.parsers import MapParser
from ..utils.collision_check import *
import numpy as np
import random

class SceneSimulator:
    def __init__(self, map_file_path, perception_range):
        self.perception_range = perception_range
        self.ego_vehicle = None
        self.agents = []
        self.map_parser = MapParser(map_file_path)
    


    def generate_traffic_agents(self, number = 0):
        while len(self.agents) < number:   
            k = random.choice(list(self.map_parser.roads))
            road = self.map_parser.roads[k]
            if (road.junction == "-1"):
                lane = random.choice(road.lanes)
                index = random.randrange(len(lane.sampled_points))
                rear_point = lane.sampled_points[index]
                hdg = lane.headings[index]
                speed = 1.0
                self.agents.append(TrafficAgent(id = str(len(self.agents)), 
                             pos=Point2D(rear_point[0], 
                                         rear_point[1]), 
                             hdg=hdg, speed=speed))
                
                

    def extract_lanes_in_range(roads, current_pos):
        cx, cy = current_pos
        for road in roads.values():
            road_in_range = False
            for lane in road.lanes:
                for (x, y) in lane.sampled_points:
                    if np.linalg.norm([x-cx, y-cy]) <= self.perception_range:
                        lane.in_range = True
                        road_in_range = True
                        break
            road.on_route = road_in_range