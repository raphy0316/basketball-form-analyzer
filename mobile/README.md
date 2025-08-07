
# Basketball Form Analyzer - Mobile App

This project is a **React Native mobile application** for analyzing basketball shooting form using computer vision. It integrates **TensorFlow Lite models** and for real-time **pose detection** and **object detection**, and communicates with a backend server for post-recording analysis and storage.

---

## 🚀 Features
- Real-time **pose detection** using MoveNet Thunder.
- **Object detection** using YOLOv8n (TFLite).
- Frame-by-frame **data collection** and visualization.
- **Backend integration** for storage and phase processing.
- Dynamic **screen rendering** with keypoints, bounding boxes, and skeleton lines.

---

## 📱 Mobile App Setup

### Requirements
- Node.js (>= 14.x)
- Yarn or npm
- React Native CLI
- Android Studio (for Android builds)
- Xcode (for iOS builds)
- Apple Developer account(free tier is fine)

### IOS setup
Open `ios/StickmanDancingApp/Info.plist` and add the required Camera Permissions:

```bash
<key>NSCameraUsageDescription</key>
<string>StickmanDancingApp needs Camera access.</string>
```

To use the CoreML GPU Delegate, add this to your `ios/Podfile`:
```bash
$EnableCoreMLDelegate=true
```
### Android setup

`TODO`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/basketball-form-analyzer.git
cd basketball-form-analyzer/mobile

# 2. Install JS dependencies
yarn install  # or npm install

# 3. Install iOS dependencies (only for macOS/iOS builds)
cd ios && pod install && cd ..

# 4. Start Metro bundler
yarn start --reset-cache
````

### Run the App

```bash
# For iOS
yarn ios

# Debug IOS
open ios/mobile.xcworkspace

# For Android
yarn android
```

---

## 🖥️ Backend Setup

The mobile app communicates with a **FastAPI** backend for pose and detection data processing.

### Requirements

* Python (>= 3.8)
* FastAPI or Flask

### Installation

```bash
# 1. Navigate to the backend folder
cd backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\\venv\\Scripts\\activate (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the backend server
cd ..
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 # in conda environment
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 # local environment
```

---

## 📁 Folder Structure

```
basketball-form-analyzer/
├── ball_extraction
├── mobile/                   # React Native frontend
├── backend/               # FastAPI/Flask backend
├── data_collection
├── data
├── pose_extration
├── phase_detection
├──basketball_shooting_analyzer.py
└── basketball_shooting_integrated_pipeline.py
```

---

## 🔗 API Endpoints

### Backend API

* `POST /model-output`: Accepts frame data with keypoints and detections
* `POST /model-output/processed`: Triggers backend analysis pipeline
* `GET /data`: Returns processed frame/shot results

---

## Notes

### Currently the target should always be at the center of the screen for model detections


Correct
```
|----------------------------------------|
|                  |||||                 |   <- Target is centered
|----------------------------------------|
```
The target (|||||) stays within the screen bounds.

Incorrect: Target Exceeds Screen Width
```
Screen Width
|----------------------------------------|
|                                ||||||||   <- Target exceeds right edge
|----------------------------------------|

Here, part of the target is **cut off** due to lack of full-screen resizing.
```


### If ever encounter the issue folly/experimental/coro/Coroutine.h' file not found on IOS build

* Please replace the Podfile with
 
```Bash
# Resolve react_native_pods.rb with node to allow for hoisting
require Pod::Executable.execute_command('node', ['-p',
  'require.resolve(
    "react-native/scripts/react_native_pods.rb",
    {paths: [process.argv[1]]},
  )', __dir__]).strip

platform :ios, '16.0'
prepare_react_native_project!
$EnableCoreMLDelegate=true

linkage = ENV['USE_FRAMEWORKS']
if linkage != nil
  Pod::UI.puts "Configuring Pod with #{linkage}ally linked Frameworks".green
  use_frameworks! :linkage => linkage.to_sym
end

target 'mobile' do
  config = use_native_modules!

  use_react_native!(
    :path => config[:reactNativePath],
    # An absolute path to your application root.
    :app_path => "#{Pod::Config.instance.installation_root}/.."
  )

  post_install do |installer|
    # https://github.com/facebook/react-native/blob/main/packages/react-native/scripts/react_native_pods.rb#L197-L202
    react_native_post_install(
      installer,
      config[:reactNativePath],
      :mac_catalyst_enabled => false,
      # :ccache_enabled => true
    )
    Dir.glob(installer.sandbox.root + "RCT-Folly/folly/Portability.h") do |file|
      begin
        # First try to make the file writable
        system "chmod +w #{file}"
        
        contents = File.read(file)
        modified_contents = contents.gsub(/#define FOLLY_HAS_COROUTINES 1/, '#define FOLLY_HAS_COROUTINES 0')
        
        # Try writing with elevated privileges if needed
        if !File.writable?(file)
          system "sudo chmod +w #{file}"
        end
        
        File.write(file, modified_contents)
        
        # Optionally restore original permissions
        system "chmod -w #{file}"
      rescue => e
        Pod::UI.warn "Failed to modify Portability.h: #{e.message}"
      end
    end
  end
end
```