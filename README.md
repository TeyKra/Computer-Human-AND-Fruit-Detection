# Human Body and Fruit Detection Project

This project showcases a dual-purpose computer vision system capable of detecting human body parts and counting apples and oranges. The project utilizes **MediaPipe** for human body part detection and **YOLOv8** for fruit detection, with the results displayed in real-time through a live camera feed.

## Project Structure

- **human.py**: This script defines the `HumanDetector` class, which uses MediaPipe for human pose estimation, hand detection, and face mesh annotation. It highlights specific body parts such as legs, arms, hands, and facial features like the eyes, nose, and mouth. Additionally, the script draws contours around the body and adds annotations for key regions.

- **fruit.py**: This script defines the `FruitDetector` class, which utilizes the pre-trained YOLOv8 model to detect fruits like apples, bananas, and oranges. The script counts the occurrences of each fruit and displays them on the video frame.

- **main.py**: This script ties everything together, allowing users to select between human detection, fruit detection, or both. It processes camera frames in real-time and displays the annotated video feed.

## Features

### Human Detection
- Detects and annotates key body parts such as shoulders, elbows, knees, and ankles.
- Draws vertical lines between the shoulders and chin to mark the neck.
- Annotates facial landmarks like the eyes, nose, mouth, and chin.

### Fruit Detection
- Detects apples and oranges using the YOLOv8 model.
- Counts the number of detected apples and oranges in real-time.
- Displays bounding boxes and labels for each detected fruit.

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install opencv-python mediapipe ultralytics numpy
```

## Usage

1. Clone the repository and navigate to the project directory:

```bash
git clone <repository_url>
cd <project_directory>
```

2. Run the `main.py` script:

```bash
python main.py
```

3. Select the mode of detection:
   - **1**: Human detection
   - **2**: Fruit detection
   - **3**: Both human and fruit detection

4. Press 'q' to quit the camera feed.

## Example Output

- In **human detection mode**, the system will annotate body parts and facial features, drawing contours and marking key areas such as shoulders, arms, legs, and feet.
- In **fruit detection mode**, it will highlight apples and oranges, showing the confidence score and a real-time count of detected fruits.

## Performance Considerations

This project is designed to run on local machines, where performance may be limited due to hardware constraints. The detection speed and accuracy may be impacted by the available computing resources. For better performance, such as faster detection and more complex models, consider deploying the system on cloud-based platforms with enhanced computational power (e.g., GPUs or TPUs). Using cloud instances will allow you to train and deploy more robust and customizable machine learning models tailored to your needs.

## Limitations

- This project is a demonstration of the potential of computer vision for real-time object detection. The fruit detection is limited to apples and oranges but can be expanded with additional training data.
- Since the script runs locally, the frame rate and detection speed might vary based on the machine's computational power.

## Future Enhancements

- Add support for more fruits by retraining the YOLOv8 model.
- Explore other pose estimation and object detection techniques to improve human detection accuracy.
- Optimize the system for cloud deployment to increase detection speed and handle larger datasets.
