import os
from ultralytics import YOLO

current_directory = os.getcwd()

# Construct full paths from the current directory
model_path = os.path.join(current_directory,'Algorithm Structure','runs', 'detect', 'train3', 'weights' , 'best.pt')
video_path = os.path.join(current_directory, '..', 'Football Analysis Proj', 'Input Videos', 'Input.mp4')

# Normalize paths
model_path = os.path.normpath(model_path)
video_path = os.path.normpath(video_path)

# Check if the paths are correct
print(f'Model path: {model_path}')
print(f'Video path: {video_path}')
print(f"Current working directory: {current_directory}")


# Verify if the model_path exists
if os.path.exists(model_path):
    print(f"File '{model_path}' exists.")
else:
    print(f"File '{model_path}' does not exist.")

# Initialize YOLO model
model = YOLO(model_path)

# Predict and save results
results = model.predict(video_path, save=True)
print(results[0])
print('=====================================')

# Print each detected box
for box in results[0].boxes:
    print(box)
