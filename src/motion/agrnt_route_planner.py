from typing import Tuple, List, Dict
import random
import logging

from ..models.agent import TrafficAgent
from ..models.map_elements import Lane

def agent_random_route(lane_changing_probability, current_agent:TrafficAgent,lanes:Dict[int,'Lane'],lanes_serch:Dict[str,'Lane'])->None:
        current_lane = lanes[current_agent.current_lane_unicode]
        if current_agent.road_map:
            hold_lane = current_agent.road_map[-1]
            pd_l = -current_agent.current_s+sum(current_agent.plan_haul)
        else:
            hold_lane = current_lane
            current_agent.road_map.append(current_lane)
            current_agent.ref_line.append(current_lane.get_ref_line())
            pd_l = current_agent.remain_s

        while pd_l<100:
                if not hold_lane.successor:
                    logging.warning(f"Agent {current_agent.id} 道路规划到断头路！")
                    break
                hold_lane_choise = random.randrange(len(hold_lane.successor))
                hold_lane = lanes_serch[hold_lane.successor[hold_lane_choise][0]+'_'+hold_lane.successor[hold_lane_choise][1]]
                current_agent.road_map.append(hold_lane)
                current_agent.ref_line.append(hold_lane.get_ref_line())
                current_agent.plan_haul.append(hold_lane.length)
                pd_l+=hold_lane.length
        
        if random.random() < lane_changing_probability and current_agent.lane_change == (-1,-1):
            current_lane = lanes[current_agent.current_lane_unicode]
            if current_agent.remain_s>max(30,2*current_agent.speed) and current_lane.belone_road.junction == "-1":
                candidate_change_lanes = []
                for lane in current_lane.belone_road.lanes:
                    if lane.lane_type==current_lane.lane_type and \
                        (int(lane.lane_id)==int(current_lane.lane_id)-1 or \
                            int(lane.lane_id)==int(current_lane.lane_id)+1):
                        candidate_change_lanes.append(lane)
                if candidate_change_lanes:
                    plan_hold_lane = random.choice(candidate_change_lanes)
                    current_agent.plan_road_map.clear()
                    current_agent.plan_ref_line.clear()
                    current_agent.plan_plan_haul.clear()
                    current_agent.lane_change = (current_lane.unicode,hold_lane.unicode)
                    plan_pd_l = current_lane.projection(current_agent.pos)[0]
                    current_agent.plan_road_map.append(plan_hold_lane)
                    current_agent.plan_ref_line.append(plan_hold_lane.get_ref_line())
                    while plan_pd_l<100:
                        if not plan_hold_lane.successor:
                            logging.warning(f"Agent {current_agent.id} 变道规划到断头路！")
                            break
                        plan_hold_lane_choise = random.randrange(len(plan_hold_lane.successor))
                        plan_hold_lane = lanes_serch[plan_hold_lane.successor[plan_hold_lane_choise][0]+'_'+plan_hold_lane.successor[plan_hold_lane_choise][1]]
                        current_agent.plan_road_map.append(plan_hold_lane)
                        current_agent.plan_ref_line.append(plan_hold_lane.get_ref_line())
                        current_agent.plan_plan_haul.append(plan_hold_lane.length)
                        plan_pd_l+=plan_hold_lane.length

            

        
