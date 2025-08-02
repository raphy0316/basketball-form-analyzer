import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

const CameraToggleButton = ({ cameraPosition, toggleCamera }) => {
  return (
    <TouchableOpacity style={styles.switchButton} onPress={toggleCamera}>
      <Text style={styles.switchButtonText}>
        Switch to {cameraPosition === 'back' ? 'Front' : 'Back'} Camera
      </Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  switchButton: {
    position: 'absolute',
    bottom: 50,
    right: 20,
    backgroundColor: '#4ECDC4',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    zIndex: 1000,
  },
  switchButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default CameraToggleButton;