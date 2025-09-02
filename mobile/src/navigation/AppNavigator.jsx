import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import LandingScreen from '../screens/LandingScreen';
// import MainScreen from '../screens/MainScreen';
import ResultsScreen from '../screens/ResultsScreen';
import ResultsDisplayScreen from '../screens/ResultsDisplayScreen';
import CameraScreen from '../screens/CameraScreen';
import PlayerSelectionScreen from '../screens/PlayerSelectionScreen';

const Stack = createStackNavigator();

const AppNavigator = () => {
  return (
    <Stack.Navigator
      initialRouteName="Landing"
      screenOptions={{
        headerShown: false,
        gestureEnabled: true,
        cardStyleInterpolator: ({ current }) => ({
          cardStyle: {
            opacity: current.progress,
          },
        }),
      }}
    >
      <Stack.Screen
        name="Landing"
        component={LandingScreen}
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="PlayerSelection"
        component={PlayerSelectionScreen}
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="Main"
        component={CameraScreen}
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="Results"
        component={ResultsScreen}
        options={{ title: 'Results' }}
      />
      <Stack.Screen
        name="ResultsDisplay"
        component={ResultsDisplayScreen}
        options={{ title: 'Results Display' }}
      />
    </Stack.Navigator>
  );
};

export default AppNavigator;