import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

import CameraScreen from './CameraScreen';
import ResultsScreen from './ResultsScreen';
import LoadingScreen from './LoadingScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Stack.Navigator
        initialRouteName="Camera"
        screenOptions={{
          headerShown: false,
        }}
      >

        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name="Results" component={ResultsScreen} />
        <Stack.Screen name="Loading" component={LoadingScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
