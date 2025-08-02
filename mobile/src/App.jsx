/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React, { useState, useEffect, performance, useRef } from 'react';
import { Frame, Camera, useCameraDevice, useFrameProcessor, runAsync} from 'react-native-vision-camera';
import {TensorflowModel, useTensorflowModel} from 'react-native-fast-tflite';

import {useResizePlugin} from 'vision-camera-resize-plugin';
import {
  Dimensions,
  StyleSheet,
  Text,
  View,
} from 'react-native'; 
import { runOnJS, useSharedValue } from 'react-native-reanimated';
import { Worklets } from 'react-native-worklets-core';
import {calcFps} from './utils/calc'
import styles from './styles/styles'
import { initCropRegion } from './utils/cropUtils';
import { applyNMS } from './utils/nmsUtils';
import Legend from './components/Legend'
import StatusOverlay from './components/StatusOverlay';
import SkeletonOverlay from './components/SkeletonOverlay';
import YoloBoxesOverlay from './components/YoloBoxesOverlay';
import KeypointOverlay from './components/KeypointOverlay';
import CameraToggleButton from './components/CameraToggleButton';

const { width: initialScreenWidth, height: initialScreenHeight } = Dimensions.get('window');

// Keypoint names for reference
const KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
];

function App() {
  const [hasPermission, setHasPermission] = useState(false);
  const [keypoints, setKeypoints] = useState([]);
  const [frameCount, setFrameCount] = useState(0);
  const [maxConfidence, setMaxConfidence] = useState(0);
  const [yoloBoxes, setYoloBoxes] = useState([]);
  const [cameraPosition, setCameraPosition] = useState('back'); // 'back' or 'front'
  // const [fps, setFps] = useState(0); // State to store FPS
  // const lastFrameTime = useRef(null); // Ref to store the last frame timestamp

  const device = useCameraDevice(cameraPosition);
  const plugin = useTensorflowModel(require('./assets/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite'), 'core-ml') // on specialized hardware
  const yoloPlugin = useTensorflowModel(require('./assets/yolov8n736-customContinue_float16.tflite'), 'core-ml'); // on specialized hardware
  // const yoloOutputRef = useSharedValue()
  const THUNDER_SIZE = 256;
  const LIGNTNING_SIZE = 192;
  const DEFAULT_FPS = 22;
  const { resize } = useResizePlugin();
  const { resize: resize2 } = useResizePlugin();
  const outputDataBuffer = [];
  // const cropRegion = useSharedValue();
  // const output = useSharedValue(0)
  const lastFrameTime = useSharedValue(0);
  const fps = useSharedValue(DEFAULT_FPS);
  const [screenWidth, setScreenWidth] = useState(initialScreenWidth);
  const [screenHeight, setScreenHeight] = useState(initialScreenHeight);
  const [screenRotation, setScreenRotation] = useState('0deg'); // Default rotation

  useEffect(() => {
    const onChange = ({ window: { width, height } }) => {
      setScreenWidth(width);
      setScreenHeight(height);
      console.log(width);
      console.log(height);
      // Determine rotation based on orientation
      if (width < height) {
        setScreenRotation('90deg'); // Landscape
      } else {
        setScreenRotation('0deg'); // Portrait
      }
    };

    const subscription = Dimensions.addEventListener('change', onChange);
    return () => subscription?.remove();
  }, []);

  // Update UI state from global variables
  useEffect(() => {
    const interval = setInterval(() => {
      const shared = global.__volatilePoseData;
      // console.log(shared)
      if (shared?.keypoints) {
        setKeypoints([...shared.keypoints]);
        setMaxConfidence(shared.confidence);
        setFrameCount(shared.frame);
        setYoloBoxes(shared.boxes);
      }
    }, 1);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    Camera.requestCameraPermission().then((p) =>
      setHasPermission(p === 'granted')
    );
  }, []);

  const toggleCamera = () => {
    setCameraPosition((prev) => (prev === 'back' ? 'front' : 'back'));
    console.log(cameraPosition)
  };

  const updatePoseData = (keypoints, confidence, frame, boxes = []) => {
    globalThis.__volatilePoseData = {
      keypoints,
      confidence,
      frame,
      boxes,
    };
  };

  const sendToBackend = async (buffer) => {
    try {
      // api call
      console.log('send to backend', buffer)
      const res = await fetch('http://192.168.3.73:8000/model-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buffer),
      });

      const json = await res.json();
      // console.log(json)
      buffer = []
      
      // console.log('Upload complete:', json);
    } catch (err) {
      console.error('Upload failed:', err.message);
    }
    outputDataBuffer.length = 0
  };
  
  const finishProcess = async () => {
    try {
      const res = await fetch('http://192.168.3.73:8000/model-output/processed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // body: JSON.stringify(buffer),
      });
      const json = await res.json();
      console.log(json);
    } catch (err) {
      console.log('finish error: ', err.message);
    }
  }

  const updateOutputData = (frameId, keypoints, detections) => {
      outputDataBuffer.push({ 
        frameId, frameWidth: screenWidth, frameHeight: screenHeight,
        keypoints, 
        detections, fps: fps.value})
    if(frameId % 20 == 0) {
      console.log('send to backend call')
      const batchToSend = [...outputDataBuffer];
      // console.log(batchToSend)
      runOnJS(sendToBackend)(batchToSend);
    }
  };
  
  // Create a worklet-safe reference
  const updatePoseDataJS = Worklets.createRunOnJS(updatePoseData);
  const updateOutputDataJS = Worklets.createRunOnJS(updateOutputData);

  const collectData = (output, yoloOutput) => {
    'worklet'
    // // MoveNet Lightning/Thunder outputs 17 keypoints * 3 values (y, x, confidence)
          const confidences = []
          const detectedKeypoints = []
          const rawKeypoints = []
          
          for (let i = 0; i < 17; i++) {
            const y = output[0][i * 3];     // Y coordinate (0-1)
            const x = output[0][i * 3 + 1]; // X coordinate (0-1)  
            const confidence = output[0][i * 3 + 2]; // Confidence score (0-1)
            confidences.push(confidence);
            const screenAspect = screenWidth / screenHeight
            // scale to fit width or height depending on screen aspect
            let scale, offsetX = 0, offsetY = 0
  
            if (screenAspect > 1) {
              // wider screen (landscape) → fit height
              scale = screenHeight
              offsetX = (screenWidth - screenHeight) / 2
            } else {
              // taller screen (portrait) → fit width
              scale = screenWidth
              offsetY = (screenHeight - screenWidth) / 2
            }

            // Apply scale and offset
            const mappedX = x * scale + offsetX
            const mappedY = y * scale + offsetY
            // Only include keypoints with reasonable confidence
            if (confidence > 0.3) {
              detectedKeypoints.push({
                id: i,
                name: KEYPOINT_NAMES[i],
                x: mappedX,
                y: mappedY,
                confidence:confidence
              });
              rawKeypoints.push({
                id: i,
                name: KEYPOINT_NAMES[i],
                x: x,
                y: y,
                confidence:confidence
              });
            }
          }  
          const predictions = [];
          const startTime = performance.now()
          // console.log(raw)
          // const keys = Object.keys(raw).sort((a, b) => parseInt(a) - parseInt(b));
          const values = yoloOutput
          // const values = raw

          const x = values.slice(0, 11109);
          const y = values.slice(11109, 11109 * 2);
          const width = values.slice(11109 * 2, 11109 * 3);
          const height = values.slice(11109 * 3, 11109 * 4);
          const class1 = values.slice(11109 * 4, 11109 * 5);
          const class2 = values.slice(11109 * 5, 11109 * 6);
          const class3 = values.slice(11109 * 6, 11109 * 7);
          if(global.frameCounter % 20 == 0)
            console.log("slicing time: ", performance.now() - startTime);
         
          for (let i = 0; i < 11109; i++) {
            const cx = x[i];
            const cy = y[i];
            const w = width[i];
            const h = height[i];
            const scores = [class1[i], class2[i], class3[i]];
            const maxConfidence = Math.max(...scores);
            const classId = scores.indexOf(maxConfidence);
            
            if (maxConfidence > 0.15  && (classId === 2 || classId === 0)) {
              predictions.push({
                x: cx - w / 2,
                y: cy - h / 2,
                width: w,
                height: h,
                confidence: maxConfidence,
                classId:classId,
              });
            }
          }
  
          // // Separate classes
          const ballCandidates = predictions.filter(b => b.classId === 0);
          // const rimCandidates = predictions.filter(b => b.classId === 2);
          // const personCandidates = predictions.filter(b => b.classId === 1);
          // Apply NMS
          const ballDetections = applyNMS(ballCandidates);
          // const rimDetections = applyNMS(rimCandidates);
          // const personDetections = applyNMS(personCandidates);
          // const parsedBoxes = []
          const parsedBoxes = ballDetections.map(box => {
            let scale, offsetX = 0, offsetY = 0;
            const screenAspect = screenWidth / screenHeight;
          
            if (screenAspect > 1) {
              // Wider screen (landscape) -> fit height
              scale = screenHeight;
              offsetX = (screenWidth - screenHeight) / 2;
            } else {
              // Taller screen (portrait) -> fit width
              scale = screenWidth;
              offsetY = (screenHeight - screenWidth) / 2;
            }

            const mappedX = box.x * scale + offsetX;
            const mappedY = box.y * scale + offsetY;
            const mappedW = box.width * scale;
            const mappedH = box.height * scale;
          
            return {
              x: mappedX,
              y: mappedY,
              width: mappedW,
              height: mappedH,
              confidence: box.confidence,
              classId: box.classId,
            };
          });

          updateOutputDataJS(global.frameCounter, detectedKeypoints, parsedBoxes); // or rawKeypoints
          updatePoseDataJS(detectedKeypoints, Math.max(...confidences), global.frameCounter, parsedBoxes);
          if(global.frameCounter % 20 == 0)
            console.log('finished frame: ', global.frameCounter)
  }

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    
    // Initialize frame counter
    if (!global.frameCounter) global.frameCounter = 0;
    global.frameCounter++;
    
    // Process every 3rd frame for performance
    // if (global.frameCounter % 3 !== 0) return;
    
    if (plugin.state === "loaded" && yoloPlugin.state === "loaded") {
      try {
        calcFps(lastFrameTime, fps);
        // if (!cropRegion.value) cropRegion.value = initCropRegion(frame.height, frame.width);
        // console.log('val', cropRegion.value)
        // const crop = {
        //   x: cropRegion.value.xMin * frame.width,
        //   y: cropRegion.value.yMin * frame.height,
        //   width: (cropRegion.value.xMax - cropRegion.value.xMin) * frame.width,
        //   height: (cropRegion.value.yMax - cropRegion.value.yMin) * frame.height,
        // };
        // console.log('crop', crop)
        const data = resize(frame, {
          scale: {
            width: THUNDER_SIZE,
            height: THUNDER_SIZE,
          },
          // crop: crop,
          pixelFormat: 'rgb',
          dataType: 'uint8',
          rotation: screenRotation
        })

        const image = resize2(frame, {
          scale: {
            width: 736,
            height: 736,
          },
          pixelFormat: 'rgb',
          dataType: 'float32',
          rotation: screenRotation
          // rotation: '90deg',
        })
        
        const startTime = performance.now();
        const output = plugin.model.runSync([data])
        // const output = []
        // console.log(output)
        if(global.frameCounter % 20 == 0)
          console.log("movenet thunder process time: ", performance.now() - startTime);
        startTime = performance.now();
        const yoloOutput = yoloPlugin.model.runSync([image]);
        // console.log(yoloOutput)
        // const yoloOutput = []
        if(global.frameCounter % 20 == 0)
          console.log("yolo process time: ", performance.now() - startTime);
        startTime = performance.now();
        const raw = yoloOutput[0]; // shape: [7, 11109]

        // if (torsoVisible(output[0])) {
        //   cropRegion.value = determineCropRegion(keypoints, frame.height, frame.width);
        // } else {
        //   cropRegion.value = initCropRegion(frame.height, frame.width);
        // }
  
        collectData(output, raw);
        if(global.frameCounter % 20 == 0)
          console.log("data processing time: ", performance.now() - startTime)
        
      } catch (error) {
        console.error(`Frame processing error: ${error.message}`);
      }
    }

  }, [plugin, yoloPlugin, screenRotation])

  return (
    <View style={StyleSheet.absoluteFill}>
      {hasPermission && device != null && (
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
          enableFpsGraph={true}
        />
      )}
     
      <SkeletonOverlay keypoints={keypoints}/>
      <YoloBoxesOverlay yoloBoxes={yoloBoxes}/>
      <KeypointOverlay keypoints={keypoints}/>
  
      {/* Status overlay */}
      <StatusOverlay
      pluginState={plugin.state}
      keypoints={keypoints}
      maxConfidence={maxConfidence}
      frameCount={global.frameCounter}
      />
      {/* Keypoint legend */}
      <Legend/>

      {!hasPermission && (
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera permission required</Text>
        </View>
      )}
      <CameraToggleButton cameraPosition={cameraPosition} toggleCamera={toggleCamera}/>
    </View>
  );
}

export default App;