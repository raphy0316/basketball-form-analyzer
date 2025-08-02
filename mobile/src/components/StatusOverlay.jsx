import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const StatusOverlay = ({ pluginState, keypoints, maxConfidence, frameCount }) => {
  return (
    <View style={styles.overlay}>
      <Text style={styles.overlayText}>Basketball Form Analyzer</Text>
      <Text style={styles.overlaySubText}>
        Model: {pluginState === 'loaded' ? '✅ Ready' : `⏳ ${pluginState || 'Loading...'}`}
      </Text>
      <Text style={styles.overlaySubText}>
        Keypoints: {keypoints.length}/17
      </Text>
      <Text style={styles.overlaySubText}>
        Max Confidence: {maxConfidence.toFixed(3)}
      </Text>
      <Text style={styles.overlaySubText}>
        Frames: {frameCount}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 50,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.8)',
    padding: 15,
    borderRadius: 8,
    minWidth: 200,
  },
  overlayText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  overlaySubText: {
    color: '#ccc',
    fontSize: 12,
    marginBottom: 4,
  },
});

export default StatusOverlay;