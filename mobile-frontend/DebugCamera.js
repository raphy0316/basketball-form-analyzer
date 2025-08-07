import React from 'react';
import { View, Text } from 'react-native';

// Try different import methods
let expoCamera = null;
let Camera = null;
let useCameraPermissions = null;

try {
  expoCamera = require('expo-camera');
  console.log('expoCamera object:', expoCamera);
  console.log('expoCamera keys:', Object.keys(expoCamera));
  
  Camera = expoCamera.Camera;
  useCameraPermissions = expoCamera.useCameraPermissions;
  
  console.log('Camera:', Camera);
  console.log('Camera type:', typeof Camera);
  console.log('useCameraPermissions:', useCameraPermissions);
  console.log('useCameraPermissions type:', typeof useCameraPermissions);
} catch (error) {
  console.error('Import error:', error);
}

const DebugCamera = () => {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Debug Camera Import</Text>
      <Text>Camera available: {Camera ? 'Yes' : 'No'}</Text>
      <Text>Camera type: {typeof Camera}</Text>
      <Text>useCameraPermissions available: {useCameraPermissions ? 'Yes' : 'No'}</Text>
    </View>
  );
};

export default DebugCamera;
