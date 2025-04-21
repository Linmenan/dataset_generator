class SceneSimulator:
    def extract_lanes_in_range(roads, current_pos, sensing_range):
        cx, cy = current_pos
        for road in roads.values():
            road_in_range = False
            for lane in road.lanes:
                for (x, y) in lane.sampled_points:
                    if np.linalg.norm([x-cx, y-cy]) <= sensing_range:
                        lane.in_range = True
                        road_in_range = True
                        break
            road.on_route = Road_In_Range