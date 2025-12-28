module.exports = function(context, options) {
  return {
    name: 'docusaurus-plugin-translation-button',

    getClientModules() {
      return [
        require.resolve('./src/components/TranslateButton/TranslationProvider'),
      ];
    },
  };
};