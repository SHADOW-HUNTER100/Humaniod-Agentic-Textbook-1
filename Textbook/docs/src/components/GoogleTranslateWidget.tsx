import React, { useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

const GoogleTranslateWidget = () => {
  return (
    <BrowserOnly>
      {() => {
        useEffect(() => {
          // Check if Google Translate script is already loaded
          if (typeof window.google === 'undefined' || typeof window.google.translate === 'undefined') {
            // Create a script element for Google Translate
            const script = document.createElement('script');
            script.type = 'text/javascript';
            script.src = '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
            document.head.appendChild(script);
          }

          // Define the initialization function
          window.googleTranslateElementInit = function() {
            if (document.getElementById('google_translate_element')) {
              new window.google.translate.TranslateElement({
                pageLanguage: 'en',
                includedLanguages: 'es,fr,de,it,pt,ur',
                layout: window.google.translate.TranslateElement.InlineLayout.SIMPLE
              }, 'google_translate_element');
            }
          };

          // Reinitialize if the function already exists but element needs to be rendered
          if (typeof window.google !== 'undefined' &&
              typeof window.google.translate !== 'undefined' &&
              document.getElementById('google_translate_element')) {
            window.googleTranslateElementInit();
          }
        }, []);

        return <div id="google_translate_element" style={{ paddingTop: '8px' }}></div>;
      }}
    </BrowserOnly>
  );
};

export default GoogleTranslateWidget;