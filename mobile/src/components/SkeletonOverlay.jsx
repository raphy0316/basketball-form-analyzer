import React from 'react';
import { View } from 'react-native';

const SKELETON = [
  ['nose', 'left_eye'],
  ['nose', 'right_eye'],
  ['left_eye', 'left_ear'],
  ['right_eye', 'right_ear'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['right_shoulder', 'right_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['right_hip', 'right_knee'],
  ['left_knee', 'left_ankle'],
  ['right_knee', 'right_ankle'],
];

const SkeletonOverlay = ({ keypoints }) => {
  return SKELETON.map((connection, index) => {
    const startPointName = connection[0];
    const endPointName = connection[1];
    
    const startPoint = keypoints.find(kp => kp.name === startPointName);
    const endPoint = keypoints.find(kp => kp.name === endPointName);
    
    if (!startPoint || !endPoint) return null;
    
    // Calculate line parameters
    const length = Math.sqrt(
      Math.pow(endPoint.x - startPoint.x, 2) + 
      Math.pow(endPoint.y - startPoint.y, 2)
    );
    
    const angle = Math.atan2(
      endPoint.y - startPoint.y,
      endPoint.x - startPoint.x
    ) * 180 / Math.PI;
    
    const avgConfidence = (startPoint.confidence + endPoint.confidence) / 2;
    
    return (
      <View
        key={`line-${index}`}
        style={{
          position: 'absolute',
          width: length,
          height: 3,
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          opacity: Math.min(avgConfidence * 2, 0.8),
          left: startPoint.x,
          top: startPoint.y,
          transformOrigin: 'left',
          transform: [
            { rotate: `${angle}deg` }
          ],
        }}
      />
    );
  });
};

export default SkeletonOverlay;