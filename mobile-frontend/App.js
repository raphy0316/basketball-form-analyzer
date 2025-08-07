import React from 'react';
import { StatusBar } from 'expo-status-bar';
import CameraScreen from './CameraScreen';

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <CameraScreen />
    </>
  );
}
