import cv2
import numpy as np
from utils.bbox_utils import get_center_of_bbox

class PlayerBallAssigner:
    def __init__(self):
        self.max_distance_threshold = 50  # Maximum distance to consider a player as interacting with the ball

    def get_distance_from_ball(self, player_bbox, ball_bbox):
        min_distance = 9999
        assigned_player_id = -1

        ball_center = get_center_of_bbox(ball_bbox)
        for player_id, player_box in player_bbox.items():
            player_center = get_center_of_bbox(player_box['bbox'])
            distance = np.sqrt((player_center[0] - ball_center[0]) ** 2 + (player_center[1] - ball_center[1]) ** 2)
            if distance < self.max_distance_threshold:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player_id = player_id 
        return assigned_player_id
    