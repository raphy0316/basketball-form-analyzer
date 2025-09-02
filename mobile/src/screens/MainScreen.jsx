// import React, { useState, useEffect, performance, useRef } from 'react';
// import { Frame, Camera, useCameraDevice, useFrameProcessor, runAsync} from 'react-native-vision-camera';
// import {TensorflowModel, useTensorflowModel} from 'react-native-fast-tflite';

// import {useResizePlugin} from 'vision-camera-resize-plugin';
// import {
//   Dimensions,
//   StyleSheet,
//   Text,
//   View,
//   TouchableOpacity
// } from 'react-native'; 
// import { useFocusEffect } from '@react-navigation/native';

// import { runOnJS, useSharedValue, runOnUI } from 'react-native-reanimated';
// import { Worklets } from 'react-native-worklets-core';
// import {calcFps} from '../utils/calc';
// import styles from '../styles/styles';
// // import { initCropRegion } from '../utils/cropUtils';
// import { applyNMS } from '../utils/nmsUtils';
// import Legend from '../components/Legend'
// import StatusOverlay from '../components/StatusOverlay';
// import SkeletonOverlay from '../components/SkeletonOverlay';
// import YoloBoxesOverlay from '../components/YoloBoxesOverlay';
// import KeypointOverlay from '../components/KeypointOverlay';
// import CameraToggleButton from '../components/CameraToggleButton';
// import StopRecordingButton from '../components/StopRecordingButton';
// import { KEYPOINT_NAMES, MIN_KEYPOINT_CONFIDENCE } from '../utils/keypointUtils';
// const { width: initialScreenWidth, height: initialScreenHeight } = Dimensions.get('window');

// const MainScreen = ({navigation}) => {
//   const [hasPermission, setHasPermission] = useState(false);
//   const [keypoints, setKeypoints] = useState([]);
//   const [frameCount, setFrameCount] = useState(0);
//   const [maxConfidence, setMaxConfidence] = useState(0);
//   const [yoloBoxes, setYoloBoxes] = useState([]);
//   const [cameraPosition, setCameraPosition] = useState('back'); // 'back' or 'front'
//   // const [isRecording, setIsRecording] = useState(false);

//   const device = useCameraDevice(cameraPosition);
//   const plugin = useTensorflowModel(require('../assets/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite'), 'core-ml') // on specialized hardware
//   const yoloPlugin = useTensorflowModel(require('../assets/yolov8n736-customContinue_float16.tflite'), 'core-ml'); // on specialized hardware
//   // const yoloOutputRef = useSharedValue()
//   const THUNDER_SIZE = 256;
//   const LIGHTNING_SIZE = 192;
//   const DEFAULT_THUNDER_FPS = 21.5;
//   const DEFAULT_LIGHTNING_FPS = 28;
//   const { resize } = useResizePlugin();
//   const { resize: resize2 } = useResizePlugin();
//   const outputDataBuffer = [];
//   // const cropRegion = useSharedValue();
//   // const output = useSharedValue(0)
//   const [screenWidth, setScreenWidth] = useState(initialScreenWidth);
//   const [screenHeight, setScreenHeight] = useState(initialScreenHeight);
//   const [screenRotation, setScreenRotation] = useState('90deg'); // Default rotation
//   const [isCameraActive, setIsCameraActive] = useState(true); // Track camera activity
//   // const isRecording = useSharedValue(false); // Shared value for communication

//   const startRecording = () => {
//     console.log('Recording started');
//     global.frameCounter = 0;
//     global.lastFrameTime = 0;
//     global.isRecording = true;
//     updateIsRecording(true);
//     setIsCameraActive(true);
//   };

//   const updateIsRecording = (newValue) => {
//     runOnUI(() => {
//       // console.log(isRecording.value);
//       global.isRecording = newValue; // Update the shared value on the Worklet thread
//       // console.log(isRecording.value);
//     })();
//   };

//   const stopRecording = async () => {
//     console.log('Recording stopped');
//     // setIsRecording(false);
//     updateIsRecording(false);
//     global.isRecording = false;
//     setIsCameraActive(false);
//     await finishProcess();
//     navigation.navigate('Results');
//   };

//   useFocusEffect(
//     React.useCallback(() => {
//       // When the screen is focused, activate the camera
//       setIsCameraActive(true);
  
//       // Reinitialize global variables
//       global.frameCounter = 0;
//       global.lastFrameTime = 0;
//       global.fps = DEFAULT_LIGHTNING_FPS;
//       global.isRecording = false;
//       return () => {
//         // When the screen loses focus, deactivate the camera and cleanup
//         setIsCameraActive(false);
//         console.log('MainScreen lost focus, stopping camera and cleanup');
  
//         // Clear global variables
//         global.frameCounter = 0
//         global.lastFrameTime = 0;
//         setFrameCount(0);
//         global.fps = DEFAULT_LIGHTNING_FPS;
//       };
//     }, [])
//   );

//   useEffect(() => {
//     const onChange = ({ window: { width, height } }) => {
//       setScreenWidth(width);
//       setScreenHeight(height);
//       console.log(width);
//       console.log(height);
//       // Determine rotation based on orientation
//       if (width < height) {
//         setScreenRotation('90deg'); // Landscape
//       } else {
//         setScreenRotation('0deg'); // Portrait
//       }
//     };

//     const subscription = Dimensions.addEventListener('change', onChange);
//     return () => subscription?.remove();
//   }, []);

//   // Update UI state from global variables
//   useEffect(() => {
//     const interval = setInterval(() => {
//       const shared = global.__volatilePoseData;
//       // console.log(shared)
//       if (shared?.keypoints) {
//         setKeypoints([...shared.keypoints]);
//         setMaxConfidence(shared.confidence);
//         setFrameCount(shared.frame);
//         setYoloBoxes(shared.boxes);
//       }
//     }, 1);

//     return () => clearInterval(interval);
//   }, []);

//   useEffect(() => {
//     Camera.requestCameraPermission().then((p) =>
//       setHasPermission(p === 'granted')
//     );
//   }, []);

//   const toggleCamera = () => {
//     setCameraPosition((prev) => (prev === 'back' ? 'front' : 'back'));
//     console.log(cameraPosition)
//   };

//   const updatePoseData = (keypoints, confidence, frame, boxes = []) => {
//     globalThis.__volatilePoseData = {
//       keypoints,
//       confidence,
//       frame,
//       boxes,
//     };
//   };

//   const sendToBackend = async (buffer) => {
//     try {
//       // api call
//       // console.log('send to backend', buffer)
//       const res = await fetch('http://192.168.3.73:8000/model-output', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(buffer),
//       });

//       const json = await res.json();
//       // console.log(json)
//       buffer = []
//       outputDataBuffer.length = 0
//       // console.log('Upload complete:', json);
//     } catch (err) {
//       console.error('Upload failed:', err.message);
//     }
//   };

//   const finishProcess = async () => {
//     try {
//       const res = await fetch('http://192.168.3.73:8000/model-output/processed', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         // body: JSON.stringify(buffer),
//       });

//       const json = await res.json();
//       console.log(json);
//       outputDataBuffer.length = 0
//       updatePoseData([], 0, 0, []);
//     } catch (err) {
//       console.log('finish error: ', err.message);
//     }
//   }

//   const stopRecord = async () => {
//     console.log('Recording stopped');
//     setIsCameraActive(false);
//     finishProcess();
//     try {
//       await finishProcess();
//     } catch (err) {
//       console.error('Error during finishProcess:', err.message);
//     }
//     global.lastFrameTime = 0;
//     global.fps = undefined;
//     console.log('Frame counter reset to:', global.frameCounter);

//     navigation.navigate('Results');    
//   };

//   const updateOutputData = (frameId, keypoints, detections, fps) => {
//       outputDataBuffer.push({ 
//         frameId, frameWidth: screenWidth, frameHeight: screenHeight,
//         keypoints, 
//         detections, fps: fps});
//     if(frameId % 20 == 0) {
//       console.log('send to backend call')
//       const batchToSend = [...outputDataBuffer];
//       // console.log(batchToSend)
//       runOnJS(sendToBackend)(batchToSend);
//     }
//   };

//   // Create a worklet-safe reference
//   const updatePoseDataJS = Worklets.createRunOnJS(updatePoseData);
//   const updateOutputDataJS = Worklets.createRunOnJS(updateOutputData);
//   const updateFrameCountJS = Worklets.createRunOnJS((frameCount) => setFrameCount);
//   // const updateFpsJS = Worklets.createRunOnJS((value) => {
//   //   setFpsDisplay(value);
//   //   // console.log(value)
//   // });
  
//   const collectData = (output, yoloOutput) => {
//     'worklet'
//     // // MoveNet Lightning/Thunder outputs 17 keypoints * 3 values (y, x, confidence)
//           const confidences = []
//           const detectedKeypoints = []
//           const rawKeypoints = []

//           for (let i = 0; i < 17; i++) {
//             const y = output[0][i * 3];     // Y coordinate (0-1)
//             const x = output[0][i * 3 + 1]; // X coordinate (0-1)  
//             const confidence = output[0][i * 3 + 2]; // Confidence score (0-1)
//             confidences.push(confidence);
//             const screenAspect = screenWidth / screenHeight
//             // scale to fit width or height depending on screen aspect
//             let scale, offsetX = 0, offsetY = 0

//             if (screenAspect > 1) {
//               // wider screen (landscape) → fit height
//               scale = screenHeight
//               offsetX = (screenWidth - screenHeight) / 2
//             } else {
//               // taller screen (portrait) → fit width
//               scale = screenWidth
//               offsetY = (screenHeight - screenWidth) / 2
//             }

//             // Apply scale and offset
//             const mappedX = x * scale + offsetX
//             const mappedY = y * scale + offsetY
//             // Only include keypoints with reasonable confidence
//             if (confidence > MIN_KEYPOINT_CONFIDENCE) {
//               detectedKeypoints.push({
//                 id: i,
//                 name: KEYPOINT_NAMES[i],
//                 x: mappedX,
//                 y: mappedY,
//                 confidence:confidence
//               });
//               rawKeypoints.push({
//                 id: i,
//                 name: KEYPOINT_NAMES[i],
//                 x: x,
//                 y: y,
//                 confidence:confidence
//               });
//             }
//           }
//           const predictions = [];
//           const startTime = performance.now()
//           // console.log(raw)
//           // const keys = Object.keys(raw).sort((a, b) => parseInt(a) - parseInt(b));
//           const values = yoloOutput
//           // const values = raw

//           const x = values.slice(0, 11109);
//           const y = values.slice(11109, 11109 * 2);
//           const width = values.slice(11109 * 2, 11109 * 3);
//           const height = values.slice(11109 * 3, 11109 * 4);
//           const class1 = values.slice(11109 * 4, 11109 * 5);
//           const class2 = values.slice(11109 * 5, 11109 * 6);
//           const class3 = values.slice(11109 * 6, 11109 * 7);
//           if(global.frameCounter % 20 == 0)
//             console.log("slicing time: ", performance.now() - startTime);

//           for (let i = 0; i < 11109; i++) {
//             const cx = x[i];
//             const cy = y[i];
//             const w = width[i];
//             const h = height[i];
//             const scores = [class1[i], class2[i], class3[i]];
//             const maxConfidence = Math.max(...scores);
//             const classId = scores.indexOf(maxConfidence);

//             if (maxConfidence > 0.15  && (classId === 2 || classId === 0)) {
//               predictions.push({
//                 x: cx - w / 2,
//                 y: cy - h / 2,
//                 width: w,
//                 height: h,
//                 confidence: maxConfidence,
//                 classId:classId,
//               });
//             }
//           }

//           // // Separate classes
//           const ballCandidates = predictions.filter(b => b.classId === 0);
//           // const rimCandidates = predictions.filter(b => b.classId === 2);
//           // const personCandidates = predictions.filter(b => b.classId === 1);
//           // Apply NMS
//           const ballDetections = applyNMS(ballCandidates);
//           // const rimDetections = applyNMS(rimCandidates);
//           // const personDetections = applyNMS(personCandidates);
//           // const parsedBoxes = []
//           const parsedBoxes = ballDetections.map(box => {
//             let scale, offsetX = 0, offsetY = 0;
//             const screenAspect = screenWidth / screenHeight;

//             if (screenAspect > 1) {
//               // Wider screen (landscape) -> fit height
//               scale = screenHeight;
//               offsetX = (screenWidth - screenHeight) / 2;
//             } else {
//               // Taller screen (portrait) -> fit width
//               scale = screenWidth;
//               offsetY = (screenHeight - screenWidth) / 2;
//             }

//             const mappedX = box.x * scale + offsetX;
//             const mappedY = box.y * scale + offsetY;
//             const mappedW = box.width * scale;
//             const mappedH = box.height * scale;

//             return {
//               x: mappedX,
//               y: mappedY,
//               width: mappedW,
//               height: mappedH,
//               confidence: box.confidence,
//               classId: box.classId,
//             };
//           });
//           updatePoseDataJS(detectedKeypoints, Math.max(...confidences), global.frameCounter, parsedBoxes);
//           if (!global.isRecording) return;
//           updateOutputDataJS(global.frameCounter, rawKeypoints, parsedBoxes, global.fps); // or rawKeypoints
          
//           if(global.frameCounter % 20 == 0)
//             console.log('finished frame: ', global.frameCounter)
//   }

//   const calcFps = () => {
//     'worklet'
//     const currentTime = performance.now();
//     // Calculate FPS
//     if (global.lastFrameTime) {
//       const deltaTime = currentTime - global.lastFrameTime;
//       global.fps = 1000 / deltaTime; // FPS = 1000ms / time between frames
//     }

//     // Update the last frame timestampr
//     global.lastFrameTime = currentTime;
//   }
  
//   const frameProcessor = useFrameProcessor((frame) => {
//     'worklet'

//     // Initialize frame counter
//     if (typeof global.frameCounter === 'undefined') {
//       global.frameCounter = 0;
//     }
//     global.frameCounter++;
//     updateFrameCountJS(global.frameCounter);
//     if(!global.fps) global.fps = DEFAULT_LIGHTNING_FPS;
//     if(!global.lastFrameTime) global.lastFrameTime = 0;
//     // Process every 3rd frame for performance
//     // if (global.frameCounter % 3 !== 0) return;
    
//     if (plugin.state === "loaded" && yoloPlugin.state === "loaded") {
//       try {
//         calcFps(global.lastFrameTime, global.fps);
    
//         // updateFpsJS(global.fps);
//         // console.log(fps.value, lastFrameTime.value)
//         // if (!cropRegion.value) cropRegion.value = initCropRegion(frame.height, frame.width);
//         // console.log('val', cropRegion.value)
//         // const crop = {
//         //   x: cropRegion.value.xMin * frame.width,
//         //   y: cropRegion.value.yMin * frame.height,
//         //   width: (cropRegion.value.xMax - cropRegion.value.xMin) * frame.width,
//         //   height: (cropRegion.value.yMax - cropRegion.value.yMin) * frame.height,
//         // };
//         // console.log('crop', crop)
//         const data = resize(frame, {
//           scale: {
//             width: LIGHTNING_SIZE,
//             height: LIGHTNING_SIZE,
//           },
//           // crop: crop,
//           pixelFormat: 'rgb',
//           dataType: 'uint8',
//           rotation: screenRotation
//         })

//         const image = resize2(frame, {
//           scale: {
//             width: 736,
//             height: 736,
//           },
//           pixelFormat: 'rgb',
//           dataType: 'float32',
//           rotation: screenRotation
//           // rotation: '90deg',
//         })
        
//         const startTime = performance.now();
//         const output = plugin.model.runSync([data])
//         // const output = []
//         // console.log(output)
//         if(global.frameCounter % 20 == 0)
//           console.log("movenet thunder process time: ", performance.now() - startTime);
//         startTime = performance.now();
//         const yoloOutput = yoloPlugin.model.runSync([image]);
//         // console.log(yoloOutput)
//         // const yoloOutput = []
//         if(global.frameCounter % 20 == 0)
//           console.log("yolo process time: ", performance.now() - startTime);
//         startTime = performance.now();
//         const raw = yoloOutput[0]; // shape: [7, 11109]

//         // if (torsoVisible(output[0])) {
//         //   cropRegion.value = determineCropRegion(keypoints, frame.height, frame.width);
//         // } else {
//         //   cropRegion.value = initCropRegion(frame.height, frame.width);
//         // }
//         console.log(global.isRecording);
       
//         collectData(output, raw);
//         if(global.frameCounter % 20 == 0)
//           console.log("data processing time: ", performance.now() - startTime)

//       } catch (error) {
//         console.error(`Frame processing error: ${error.message}`);
//       }
//     }

//   }, [plugin, yoloPlugin, screenRotation])

//   return (
//     <View style={StyleSheet.absoluteFill}>
//       {hasPermission && device != null && (
//         <Camera
//           style={StyleSheet.absoluteFill}
//           device={device}
//           isActive={isCameraActive}
//           frameProcessor={frameProcessor}
//           pixelFormat="yuv"
//           enableFpsGraph={true}
//         />
//       )}

//       <SkeletonOverlay keypoints={keypoints}/>
//       <YoloBoxesOverlay yoloBoxes={yoloBoxes}/>
//       <KeypointOverlay keypoints={keypoints}/>

//       {/* Status overlay */}
//       <StatusOverlay
//       pluginState={plugin.state}
//       keypoints={keypoints}
//       maxConfidence={maxConfidence}
//       frameCount={frameCount}
//       />
//       {/* Keypoint legend */}
//       <Legend/>

//       {!hasPermission && (
//         <View style={styles.permissionContainer}>
//           <Text style={styles.permissionText}>Camera permission required</Text>
//         </View>
//       )}
//       <CameraToggleButton cameraPosition={cameraPosition} toggleCamera={toggleCamera}/>
//       {/* <StopRecordingButton onStopRecording={stopRecording} /> */}
//       <View style={{ position: 'absolute', bottom: 40, alignSelf: 'center', flexDirection: 'row', gap: 20 }}>
//         <TouchableOpacity
//           style={{ backgroundColor: 'red', paddingVertical: 12, paddingHorizontal: 24, borderRadius: 8 }}
//           onPress={startRecording}
//         >
//           <Text style={{ color: 'white', fontSize: 16 }}>Start Recording</Text>
//         </TouchableOpacity>

//         <TouchableOpacity
//           style={{ backgroundColor: 'green', paddingVertical: 12, paddingHorizontal: 24, borderRadius: 8 }}
//           onPress={stopRecording}
//         >
//           <Text style={{ color: 'white', fontSize: 16 }}>Stop Recording</Text>
//         </TouchableOpacity>
//       </View>
//     </View>
//   );
// }

// export default MainScreen;
