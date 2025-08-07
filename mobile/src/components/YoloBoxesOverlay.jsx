import React from 'react';
import { View, Text } from 'react-native';

const YoloBoxesOverlay = ({ yoloBoxes }) => {
  return yoloBoxes.map((box, index) => (
    <View
      key={`box-${index}`}
      style={{
        position: 'absolute',
        left: box.x,
        top: box.y,
        width: box.width,
        height: box.height,
        borderWidth: 2,
        borderColor: 'lime',
        backgroundColor: 'rgba(0,255,0,0.1)',
      }}
    >
      <Text
        style={{
          color: 'white',
          backgroundColor: 'black',
          fontSize: 10,
          position: 'absolute',
          top: -14,
          left: 0,
          paddingHorizontal: 4,
        }}
      >
        {`Class ${box.classId} (${(box.confidence * 100).toFixed(0)}%)`}
      </Text>
    </View>
  ));
};

export default YoloBoxesOverlay;