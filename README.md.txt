[12:06 pm, 18/9/2025] £€©J: # Fall Detection and Alert System

This system uses computer vision and machine learning to detect falls based on body postures and sends alerts via SMS and webhooks.

## Setup Instructions

1. Install Dependencies:
 2. Set up Twilio for SMS Alerts* (Optional):
- Create a Twilio account at https://www.twilio.com/
- Update twilio_account_sid, twilio_auth_token, and phone numbers in the FallDetector class

3. Set up Webhook Endpoint (Optional):
- Update webhook_url in the FallDetector class to your endpoint

4. Prepare Dataset:
- Run the program and choose option 1 to capture images from your webcam
- Or manually place images in respective folders under dataset/:
  - dataset/standing/
  - dataset/sitting/
  - dataset/bending/
  - dataset/falling/
  - dataset/tripping/

5. Train the Model:
- Run the program and choose option 2 to train the posture classification model

6. Run the Application:
- Execute the program and choose option 3 to start the fall detection system

# Usage

1. The system will access your webcam (change source if needed)
2. It will classify postures in real-time
3. If a fall pattern is detected, it will send alerts via SMS and webhook
4. Press 'q' to quit the application

# Customization

- Adjust alert_cooldown to change how often alerts can be sent
- Modify the fall detection logic in check_for_fall() method
- Add more posture classes by creating additional folders in the dataset
 
#file structure

fall-detection-system/
├── dataset/ # Training images by category
├── models/ # Saved models and training history
├── test_images/ # Images for testing
├── fall_detection.py # Main application file
├── requirements.txt # Python dependencies
└── README.md # This file