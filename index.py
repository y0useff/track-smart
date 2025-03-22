from roboflow import Roboflow
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Get video capture
video = cv2.VideoCapture("clip1.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Setup video writer for output (a new file to avoid overwriting the input file)
out = cv2.VideoWriter('output_with_quadrilateral.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Define custom quadrilateral coordinates
quad_points = [
    (0, 1550),  # top-left
    (2500, 650),  # top-right
    (2500, 750),  # bottom-right
    (0, 1250)  # bottom-left
]

# Process video frames
while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    # Draw quadrilateral on the frame
    for i in range(4):
        cv2.line(frame, quad_points[i], quad_points[(i+1)%4], (0, 0, 255), 2)
    
    # Draw diagonals of the quadrilateral
    cv2.line(frame, quad_points[0], quad_points[2], (0, 0, 255), 2)
    cv2.line(frame, quad_points[1], quad_points[3], (0, 0, 255), 2)

    # Display the frame with quadrilateral
    cv2.imshow("Tracking with Custom Boundary", frame)
    
    # Write the frame with quadrilateral to the output file
    out.write(frame)

    # Allow user to quit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
video.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Now run YOLO tracking on the saved video with quadrilateral
results = model.track(source="output_with_quadrilateral.mp4", show=True, tracker="bytetrack.yaml", save=True)
