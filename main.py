from Algorithm_Structure.utils import read_video, save_video, test
from Algorithm_Structure.tracking.tracker import Tracker
from Algorithm_Structure.color_classifier import TeamAssigner
from Algorithm_Structure.player_ball_assigner import PlayerBallAssigner
from Algorithm_Structure.camera_movement_estimator import CameraMovementEstimator
from Algorithm_Structure.prespective_transformation import ViewTransformer
from Algorithm_Structure.speed_distance_estimator import SpeedAndDistance_Estimator
import numpy as np
import cv2
if __name__ == "__main__":
    video_path = r'Input Videos\Input.mp4'
    output_path = r'Output Videos\Output.mp4'
    model_path = r"Algorithm_Structure\runs\detect\train3\weights\best.pt"
    video_frames = read_video(video_path)

    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

     # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # perspective transformation
    perspective_transformer = ViewTransformer()
    perspective_transformer.add_transformed_position_to_tracks(tracks)
    #Heat map 
    tracker.collect_pitch_positions(tracks)  # collect all players' pitch positions

  # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

     # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.get_distance_from_ball(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    output_video_frames = tracker.draw_annotations(video_frames,tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames,
        tracks
    )
    # For now just save original frames
    save_video(output_video_frames, output_path)

    # heatmap
    # for player_id in tracker.player_positions.keys():
    #     heatmap = tracker.generate_player_heatmap(player_id)

    #     if heatmap is not None:
    #     # Save the heatmap as an image
    #         cv2.imwrite(f"../heatmap_players/heatmap_player_{player_id}.png", heatmap)
    print("Done")
