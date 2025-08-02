import React from 'react';
import { View, Text } from 'react-native';
import styles from '../styles/styles';

const KeypointOverlay = ({ keypoints }) => {
  const getKeypointColor = (name) => {
    if (name.includes('eye') || name.includes('ear') || name === 'nose')
      return '#FF6B6B'; // Red for face
    if (name.includes('shoulder') || name.includes('elbow') || name.includes('wrist'))
      return '#4ECDC4'; // Teal for arms
    if (name.includes('hip') || name.includes('knee') || name.includes('ankle'))
      return '#45B7D1'; // Blue for legs
    return '#96CEB4'; // Default green
  };

  return keypoints.map((keypoint) => (
    <View
      key={keypoint.id}
      style={[
        styles.keypoint,
        {
          left: keypoint.x - 8,
          top: keypoint.y - 8,
          backgroundColor: getKeypointColor(keypoint.name),
          opacity: Math.min(keypoint.confidence * 3, 1),
        },
      ]}
    >
      <Text style={styles.keypointText}>{keypoint.id}</Text>
    </View>
  ));
};

export default KeypointOverlay;