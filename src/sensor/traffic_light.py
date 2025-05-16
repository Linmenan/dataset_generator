from typing import Tuple, List

from ..models.map_elements import Lane,Controller

def get_lane_traffic_light(lane: Lane , controlls: List[Controller], sim_time:float) -> Tuple[str,float,str]:
    road = lane.belone_road
    light = ('grey',0.0,'')
    if road.junction != "-1":
            if road.signals is not None:
                for signal in road.signals:
                    if signal.type=='reference':
                        id = signal.signal_id
                        from_lane = int(signal.from_lane)
                        to_lane =  int(signal.to_lane)
                        turn_relation = signal.turn_relation
                        if int(lane.lane_id)>=from_lane and int(lane.lane_id)<=to_lane:
                            for controll in controlls:
                                for control in controll.controls:
                                    if control.signal_id == id:
                                        if controll.signal_controller is not None:
                                            status = controll.signal_controller.state_with_countdown(t=sim_time)
                                            now_ = status[id]
                                            light = (now_[0].value,now_[1],turn_relation)
    return light