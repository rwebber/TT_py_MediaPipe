import cv2
import mediapipe as mp
import json
import numpy as np


"""
===============================================================================
MediaPipe Pose → JSON Output Script for Isadora Pythoner
===============================================================================

OVERVIEW
--------
This script captures pose landmarks from a video frame using Google's
MediaPipe Pose solution, and outputs a JSON record containing the 33 standard
pose landmarks (x, y, z, visibility) for use within Isadora.

The output JSON structure is:
    {
      "pose": [
        { "id": 0, "name": "Nose", "x": 0.71, "y": 0.60, "z": -0.72, "visibility": 0.99 },
        ...
      ]
    }

Each landmark includes:
  • id          → the landmark index (0–32)
  • name        → the human-readable joint name
  • x, y, z     → normalized landmark coordinates (range ~0–1)
  • visibility  → landmark confidence (0–1)

NOTE: This JSON can be converted to Isadora’s Skeleton JSON format using a
companion User Actor "DX - MediaPipe to Izzy Skeleton", which is present in the example file.

-------------------------------------------------------------------------------
INCLUDED FUNCTIONALITY
-------------------------------------------------------------------------------
  ✓ Uses the MediaPipe Pose model with:
      - min_detection_confidence = 0.5
      - min_tracking_confidence  = 0.5
  ✓ Processes incoming frames as NumPy arrays (e.g., from Isadora’s live feed)
  ✓ Converts BGR → RGB for MediaPipe
  ✓ Returns both JSON data and a status message
  ✓ Includes built-in camera test block when run standalone (for debugging)
  ✓ Proper initialization and cleanup of the MediaPipe Pose instance

-------------------------------------------------------------------------------
EXCLUDED / OPTIONAL FEATURES (AVAILABLE IN MEDIAPIPE)
-------------------------------------------------------------------------------
  ✗ pose_world_landmarks
      - Provides 3D world coordinates in meters.
      - Useful for metric distance or angle measurements.
      - Can be added via: results.pose_world_landmarks.landmark

  ✗ segmentation_mask
      - Produces a per-pixel person mask for background separation.
      - Enable via: Pose(enable_segmentation=True, smooth_segmentation=True)
      - Returns: results.segmentation_mask (NumPy array)

  ✗ pixel coordinate conversion
      - The current output uses normalized coordinates (0–1).
      - Pixel coordinates can be added via (x_px = x * width, y_px = y * height), generally not needed in Isadora

  ✗ multiple pose tracking
      - MediaPipe Pose currently tracks a single person per frame.
      - ! tracking multiple people was note possible (in development) at the time of this scripts creation

-------------------------------------------------------------------------------
OUTPUT DETAILS
-------------------------------------------------------------------------------
  Output 1 (pose_json_output):  JSON string containing the landmark data.
  Output 2 (status_message):     Human-readable message for logging or debugging.

-------------------------------------------------------------------------------
NOTES
-------------------------------------------------------------------------------
  • All coordinates are normalized to the input frame dimensions.
  • Y values increase downward (image coordinates), consistent with MediaPipe.
  • This script can run standalone for testing, capturing from the default webcam.
  • Designed for use inside Isadora’s Pythoner actor but portable to any Python environment.

===============================================================================
"""


# iz_input 1 "video_frame"
# iz_output 1 "pose_json_output"
# iz_output 2 "status_message"

# Names are not provided by default, so we use a dictionary to map the names making the output human readable.
POSE_LANDMARKS = {
    0: "Nose", 1: "Left Eye (Inner)", 2: "Left Eye", 3: "Left Eye (Outer)",
    4: "Right Eye (Inner)", 5: "Right Eye", 6: "Right Eye (Outer)",
    7: "Left Ear", 8: "Right Ear", 9: "Mouth (Left)", 10: "Mouth (Right)",
    11: "Left Shoulder", 12: "Right Shoulder", 13: "Left Elbow", 14: "Right Elbow",
    15: "Left Wrist", 16: "Right Wrist", 17: "Left Pinky", 18: "Right Pinky",
    19: "Left Index Finger", 20: "Right Index Finger", 21: "Left Thumb", 22: "Right Thumb",
    23: "Left Hip", 24: "Right Hip", 25: "Left Knee", 26: "Right Knee",
    27: "Left Ankle", 28: "Right Ankle", 29: "Left Heel", 30: "Right Heel",
    31: "Left Foot Index", 32: "Right Foot Index"
}

pose = None  # Global variable to hold the MediaPipe Pose instance

# python_init is called when the actor is first activated.
def python_init(video_frame):
    """
    Initializes a MediaPipe Pose detection instance with the given parameters.
    This function sets up necessary configurations for pose estimation.

    :param video_frame: The input video frame used for processing with Pose.
                        This parameter is not directly used in the initialization process.
    :type video_frame: Any
    :return: None
    :rtype: None
    """
    global pose
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return  # No additional initialization required

# python_main is called whenever an input value changes
def python_main(video_frame):
    """
    Processes a given video frame to extract pose landmarks using Mediapipe's pose
    detection functionality. If the input frame is not valid or an error occurs
    during processing, appropriate error messages are returned.

    :param video_frame: A single video frame represented as a NumPy image array.
    :type video_frame: numpy.ndarray
    :return: A tuple containing a JSON string with detected pose landmarks or an
        error message, and a descriptive status message.
    :rtype: (str, str)
    """
    if not isinstance(video_frame, np.ndarray) or video_frame.size == 0:
        return json.dumps({"error": "Invalid input: Expected a non-empty NumPy image array"}), "Invalid or empty frame received"

    try:
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
    except cv2.error as e:
        return json.dumps({"error": "OpenCV error: " + str(e)}), "OpenCV processing error"

    # Extract pose landmarks
    if results.pose_landmarks:
        landmarks = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                "id": idx,
                "name": POSE_LANDMARKS[idx],
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        return json.dumps({"pose": landmarks}), "Pose detected successfully"
    else:
        return json.dumps({"pose": []}), "No pose detected"

# python_finalize is called just before the actor is deactivated
def python_finalize():
    global pose
    if pose:
        pose.close()
        pose = None
    return

if __name__ == "__main__":
    """ 
    This section is used to call the python_main() function from an IDE
    in a similar way as it will be called in Pythoner.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera to supply livefeed to pose detection. 
    # NOTE: this is like using Isadoras' live feed and linking it to a video input in the Pythoner actor
    if not cap.isOpened():
        print("Error: Could not open video stream.")
    else:
        try:
            python_init(None)  # Initialize MediaPipe Pose
            while True:
                # Below, reading the capured video frame returns two values, the first 'ret' allows us to know true/false for the success of the capture
                # the second 'frame' value contains the image data if available.
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Status: No valid frame received")
                    continue
                # python_main() defined above, returns two values, first the POSE json data if available, and second a status message
                output_json, status_message = python_main(frame)
                # we print these values so that we can see the output, similarly to how they would be output from Pythoner.
                print(output_json)
                print("Status:", status_message)
        except KeyboardInterrupt:
            pass  # Stop on user interrupt (Ctrl+C)
        finally:
            cap.release()
            python_finalize()
