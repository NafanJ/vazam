module.exports = {
  root: true,
  extends: '@react-native',
  rules: {
    // Formatting is the editor's job; lint enforces correctness only.
    'prettier/prettier': 'off',
    // This codebase uses double quotes throughout.
    quotes: 'off',
  },
};
