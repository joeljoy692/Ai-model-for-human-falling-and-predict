from dotenv import load_dotenv
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import random
import urllib.request
import zipfile
from twilio.rest import Client
import requests
load_dotenv()
# Access your secrets using os.getenv
twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_token = os.getenv('TWILIO_AUTH_TOKEN')

# Create directory structure
os.makedirs('dataset/standing', exist_ok=True)
os.makedirs('dataset/sitting', exist_ok=True)
os.makedirs('dataset/bending', exist_ok=True)
os.makedirs('dataset/falling', exist_ok=True)
os.makedirs('dataset/tripping', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('test_images', exist_ok=True)
os.makedirs('public_datasets', exist_ok=True)

def capture_images(label, num_images=50):
    """
    Capture images from webcam for different postures
    """
    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"Capturing {num_images} images for {label}. Press 's' to start, 'q' to quit")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, f"Capturing: {label} ({count}/{num_images})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Capturing Postures', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            
            img_name = f"dataset/{label}/{label}_{count}.jpg"
            cv2.imwrite(img_name, frame)
            count += 1
            print(f"Saved {img_name}")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def download_ur_fall_dataset():
    """
    Download UR Fall Detection dataset
    """
    print("Downloading UR Fall Detection Dataset...")
    url = "http://fenix.univ.rzeszow.pl/~mkepski/ds/data/urfall-cam0.zip"
    zip_path = "public_datasets/ur_fall_dataset.zip"
    extract_path = "public_datasets/ur_fall_dataset"
    
    # Download the dataset
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete!")
    
    # Extract the dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete!")
    
    return extract_path

def extract_frames_from_video(video_path, output_folder, frames_per_second=1):
    """
    Extract frames from a video file
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second) if fps > 0 else 1
    count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
        count += 1
        
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

def organize_ur_dataset(dataset_path):
    """
    Organize the UR Fall Detection dataset into our folder structure
    """
    # This is a simplified organization - you might need to adjust based on the actual dataset structure
    adl_path = os.path.join(dataset_path, "adl")
    fall_path = os.path.join(dataset_path, "fall")
    
    # Process activities of daily living (non-fall)
    if os.path.exists(adl_path):
        for video_file in os.listdir(adl_path):
            if video_file.endswith('.avi'):
                video_path = os.path.join(adl_path, video_file)
                
                extract_frames_from_video(video_path, "dataset/standing", 2)
    
    # Process fall videos
    if os.path.exists(fall_path):
        for video_file in os.listdir(fall_path):
            if video_file.endswith('.avi'):
                video_path = os.path.join(fall_path, video_file)
                extract_frames_from_video(video_path, "dataset/falling", 2)

def create_model(input_shape=(128, 128, 3), num_classes=5):
    """
    Create a CNN model for posture classification
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def preprocess_data(data_dir='dataset', img_height=128, img_width=128):
    """
    Preprocess images and prepare for training
    """
    # Check if dataset directory exists and has subdirectories
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: No data found in {data_dir}. Please add images to the subdirectories first.")
        return None, None, None
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=32)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=32)
    
    class_names = train_ds.class_names
    print("Class names:", class_names)
    
    # Cache and prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names

def train_model():
    """
    Train the posture classification model
    """
    train_ds, val_ds, class_names = preprocess_data()
    
    if train_ds is None:
        print("Cannot train model without data. Please add images to the dataset folders first.")
        return None, None
    
    model = create_model()
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('models/posture_model.h5', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )
    
    # Save the class names
    with open('models/class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.savefig('models/training_history.png')
    plt.show()
    
    return model, class_names

class FallDetector:
    def __init__(self, model_path='models/posture_model.h5'):
        # Load model if it exists, otherwise create a placeholder
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            with open('models/class_names.txt', 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            print("Model not found. Using placeholder classifier.")
            self.model = None
            self.class_names = ['standing', 'sitting', 'bending', 'falling', 'tripping']
        
        # State variables for tracking patterns
        self.posture_history = []
        self.fall_detected = False
        self.alert_sent = False
        self.alert_cooldown = 30  # seconds between alerts
        self.last_alert_time = 0
        
        # Twilio configuration (replace with your actual credentials)
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')9'
        self.twilio_phone_number = '+14787804623'
        self.emergency_contact = '+916238415056'
        
        # Web server endpoint for alerts
        self.webhook_url = 'https://your-webhook-endpoint.com/alert'
    
    def detect_posture(self, frame):
        """
        Detect posture from a frame
        """
        if self.model is None:
            
            return "standing", 0.8  
        
        # Preprocess frame
        img = cv2.resize(frame, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        posture = self.class_names[predicted_class]
        
        return posture, confidence
    
    def check_for_fall(self, current_posture, confidence):
        """
        Check if a fall has occurred based on posture history
        """
        # Add to history
        self.posture_history.append((current_posture, confidence, time.time()))
        
        # Keep only recent history (last 10 seconds)
        current_time = time.time()
        self.posture_history = [(p, c, t) for p, c, t in self.posture_history 
                               if current_time - t < 10]
        
        # Check for sudden change from standing to falling
        if len(self.posture_history) >= 3:
            # Look for pattern: standing -> (bending/tripping) -> falling
            recent_postures = [p for p, c, t in self.posture_history[-3:]]
            
            if (recent_postures[-1] == 'falling' and 
                recent_postures[-2] in ['bending', 'tripping'] and 
                recent_postures[-3] == 'standing'):
                return True
        
        return False
    
    def send_alert(self, reason="Fall detected"):
        """
        Send alert via SMS and webhook
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"ALERT: {reason} at {current_time}. Immediate attention required!"
        
        # Send SMS via Twilio (if configured)
        if (self.twilio_account_sid != 'your_account_sid' and 
            self.twilio_auth_token != 'your_auth_token'):
            try:
                client = Client(self.twilio_account_sid, self.twilio_auth_token)
                message = client.messages.create(
                    body=message,
                    from_=self.twilio_phone_number,
                    to=self.emergency_contact
                )
                print(f"SMS alert sent: {message.sid}")
            except Exception as e:
                print(f"Failed to send SMS: {e}")
        else:
            print("Twilio not configured. Skipping SMS alert.")
        
        # Send webhook alert (if configured)
        if self.webhook_url != 'https://your-webhook-endpoint.com/alert':
            try:
                payload = {
                    "timestamp": current_time,
                    "alert_type": reason,
                    "message": message
                }
                response = requests.post(self.webhook_url, json=payload)
                print(f"Webhook alert sent: {response.status_code}")
            except Exception as e:
                print(f"Failed to send webhook alert: {e}")
        else:
            print("Webhook not configured. Skipping webhook alert.")
        
        self.alert_sent = True
        self.last_alert_time = time.time()
    
    def process_video_stream(self, source=0):
        """
        Process video stream for real-time fall detection
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting video processing. Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect posture
            posture, confidence = self.detect_posture(frame)
            
            # Check for fall
            if self.check_for_fall(posture, confidence):
                if not self.fall_detected:
                    self.fall_detected = True
                    print("FALL DETECTED!")
                    
                    # Check if we need to send an alert
                    current_time = time.time()
                    if (not self.alert_sent or 
                        (self.alert_sent and current_time - self.last_alert_time > self.alert_cooldown)):
                        self.send_alert(f"Fall detected with confidence {confidence:.2f}")
            else:
                self.fall_detected = False
            
            # Display results
            cv2.putText(frame, f"Posture: {posture} ({confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.fall_detected:
                cv2.putText(frame, "FALL DETECTED!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.imshow('Fall Detection System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def test_with_images(image_dir='test_images'):
    """
    Test the model with sample images
    """
    detector = FallDetector()
    
    # Check if test images directory exists
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f"No test images found in {image_dir}. Please add some test images first.")
        return
    
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            
            if image is not None:
                posture, confidence = detector.detect_posture(image)
                
                # Display result
                plt.figure(figsize=(8, 6))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f"Predicted: {posture} ({confidence:.2f})")
                plt.axis('off')
                plt.show()
                
                print(f"Image: {img_file}, Posture: {posture}, Confidence: {confidence:.2f}")

def main():
    """
    Main function to run the fall detection system
    """
    while True:
        print("\nFall Detection and Alert System")
        print("1. Capture training images")
        print("2. Download UR Fall Detection dataset")
        print("3. Train model")
        print("4. Run fall detection")
        print("5. Test with images")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            label = input("Enter posture label (standing, sitting, bending, falling, tripping): ")
            num_images = int(input("Number of images to capture (default 50): ") or "50")
            capture_images(label, num_images)
        elif choice == "2":
            print("Downloading UR Fall Detection Dataset...")
            try:
                dataset_path = download_ur_fall_dataset()
                print("Organizing dataset...")
                organize_ur_dataset(dataset_path)
                print("Dataset downloaded and organized!")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
        elif choice == "3":
            print("Training model...")
            try:
                model, class_names = train_model()
                if model is not None:
                    print("Model trained successfully!")
            except Exception as e:
                print(f"Error training model: {e}")
        elif choice == "4":
            print("Initializing fall detector...")
            detector = FallDetector()
            print("Starting fall detection...")
            detector.process_video_stream()  # 0 for default camera
        elif choice == "5":
            print("Testing with images...")
            test_with_images()
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()