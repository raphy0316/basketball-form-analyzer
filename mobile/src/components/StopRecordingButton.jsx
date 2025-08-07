import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

const StopRecordingButton = ({ onStopRecording }) => {
  return (
    <TouchableOpacity style={styles.button} onPress={onStopRecording}>
      <Text style={styles.buttonText}>Stop Recording</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    position: 'absolute',
    bottom: 50,
    right: 20,
    backgroundColor: '#FF6B6B',
    paddingVertical: 10,
    paddingHorizontal: 26,
    borderRadius: 8,
    zIndex: 1000,
  },
  buttonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
});

export default StopRecordingButton;