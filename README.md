# Privacy-First AI App: Real-Time Face Detection (YOLOv8), Age Classification, & Child-Face Censoring
A practical, privacy-first mobile application that performs real-time face detection and age classification directly on your device. This project demonstrates how to overcome the key engineering challenges of running complex AI models on mobile platforms while maintaining a smooth, responsive user experience.

🌟 Key Features
🚀 Fast Face Detection: Utilizes a lightweight YOLOv8s-face model to instantly identify faces in any image.

🤖 Accurate Age Classification: A custom-trained classifier classifies age range of the detected faces.

🔒 100% On-Device Processing: All AI computations happen on your phone. No data is ever sent to the cloud, ensuring absolute privacy.

✈️ Works Offline: The app is fully functional without an internet connection, making it reliable anytime, anywhere.

🧈 Smooth & Responsive UI: Built from the ground up to ensure that intensive AI tasks do not freeze the user interface.

🤔 The Challenge: AI on Mobile is Hard
Putting a powerful AI model into a mobile app isn't as simple as just dropping the model file into the project. We had to overcome two major hurdles unique to mobile devices.

1. The On-Device AI Challenge
Smartphones, while powerful, are not supercomputers. Unlike AI systems that run on massive cloud servers, a mobile app has to work within tight constraints:

Processing Power: Running a deep learning model is computationally expensive and can be slow on a phone's processor.

Memory (RAM): Large models and high-resolution images can quickly consume a device's limited RAM, leading to crashes.

Battery Life: AI processing is energy-intensive and can significantly drain a phone's battery.

A key part of this project was engineering a way to run our models efficiently without overwhelming the device.

2. The UI Responsiveness Challenge
A mobile app's interface runs on a single "UI thread." This thread handles everything you see and touch—animations, scrolling, and button taps. If a heavy task, like our AI processing, runs on this thread, the entire app freezes. It becomes unresponsive and feels broken.

A central engineering problem was figuring out how to perform these intensive AI calculations in the background, ensuring the app remains perfectly smooth at all times.

💡 Our Solution & Philosophy
This project's goal wasn't just to build another AI model, but to engineer a solution that is practical, accessible, and respectful of user privacy.

This was achieved through a series of deliberate trade-offs and technical decisions:

Prioritizing User Experience Over Brute Force: Instead of using a large, slow, but marginally more accurate model, this project combines a lightweight YOLOv8s-face model with a focused, custom-trained age classifier. This conscious trade-off sacrifices a tiny amount of analytical power for a massive gain in speed, providing results in seconds, not minutes.

On-Device Processing for Privacy: Many AI solutions are cloud-based, requiring users to upload personal photos to a server. This raises significant privacy concerns. By ensuring all processing happens locally, this app is fundamentally more private and secure.

🚀 Getting Started
(This is a placeholder section. You can update it with your specific build instructions.)

To get a local copy up and running, follow these simple steps.

Prerequisites
Xcode / Android Studio

Swift / Kotlin environment

Installation
Clone the repo

git clone [https://github.com/Srophos/face_censoring_AI_app](github.com/Srophos/face_censoring_AI_app)

Open the project in your IDE.

Install dependencies (e.g., using CocoaPods or Gradle).

Build and run the application on a physical device.

🖼️ Usage


Launch the app.

Grant camera or photo library permissions.

Select an image from your gallery or take a new photo.

The app will automatically detect faces and display the range of their age. In case of children from the age of 1-13 years it will automatically prompt the user if they want to censor their faces and show the preview.
