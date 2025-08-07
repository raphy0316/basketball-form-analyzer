

import React, { useState, useEffect, performance, useRef } from 'react';

const calcFps = (lastFrameTime, fps) => {
    'worklet'
    const currentTime = performance.now();
    // Calculate FPS
    if (lastFrameTime.value) {
      const deltaTime = currentTime - lastFrameTime.value;
      fps.value = 1000 / deltaTime; // FPS = 1000ms / time between frames
    }

    // Update the last frame timestampr
    lastFrameTime.value = currentTime;
  }

export {calcFps};