module.exports = {
  presets: [
    'module:@react-native/babel-preset',
    '@babel/preset-typescript'
  ],
  plugins: [
    '@babel/plugin-proposal-optional-chaining',
    '@babel/plugin-proposal-nullish-coalescing-operator',
    '@babel/plugin-transform-optional-chaining',
    '@babel/plugin-transform-nullish-coalescing-operator',
    ['react-native-worklets-core/plugin'],
    // 'react-native-reanimated/plugin',
    ["react-native-reanimated/plugin", { processNestedWorklets: true }],
  ],
};