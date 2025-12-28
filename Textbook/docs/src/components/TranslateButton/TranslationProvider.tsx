import React, { useEffect, useState } from 'react';

const TranslationProvider: React.FC = () => {
  const [isTranslated, setIsTranslated] = useState(false);
  const [originalContent, setOriginalContent] = useState<Record<string, string>>({});

  useEffect(() => {
    const addTranslationButton = () => {
      // Wait for the DOM to be ready
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectButton);
      } else {
        injectButton();
      }
    };

    const injectButton = () => {
      // Remove existing button if present
      const existingButton = document.getElementById('translation-button-container');
      if (existingButton) {
        existingButton.remove();
      }

      // Create the translation button
      const buttonContainer = document.createElement('div');
      buttonContainer.id = 'translation-button-container';
      buttonContainer.style.cssText = `
        margin-bottom: 20px;
        text-align: center;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
      `;

      const button = document.createElement('button');
      button.id = 'translate-to-urdu';
      button.textContent = isTranslated ? 'Show Original' : 'Translate to Urdu';
      button.style.cssText = `
        background: ${isTranslated
          ? 'linear-gradient(90deg, #ea4335, #fbbc05, #34a853, #4285f4)'
          : 'linear-gradient(90deg, #4285f4, #34a853, #fbbc05, #ea4335)'};
        color: white;
        border: none;
        padding: 12px 24px;
        cursor: pointer;
        border-radius: 24px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
      `;

      button.addEventListener('mouseover', function() {
        (this as HTMLElement).style.transform = 'translateY(-2px)';
        (this as HTMLElement).style.boxShadow = '0 6px 16px rgba(0,0,0,0.3)';
      });

      button.addEventListener('mouseout', function() {
        (this as HTMLElement).style.transform = 'translateY(0)';
        (this as HTMLElement).style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
      });

      button.addEventListener('click', async () => {
        if (isTranslated) {
          // Show original content
          restoreOriginalContent();
          setIsTranslated(false);
          button.textContent = 'Translate to Urdu';
          button.style.background = 'linear-gradient(90deg, #4285f4, #34a853, #fbbc05, #ea4335)';
        } else {
          // Translate content
          await translatePageContent();
          setIsTranslated(true);
          button.textContent = 'Show Original';
          button.style.background = 'linear-gradient(90deg, #ea4335, #fbbc05, #34a853, #4285f4)';
        }
      });

      buttonContainer.appendChild(button);

      // Try to find the main content area and inject the button at the beginning
      const selectors = [
        '[data-testid="doc-root"]',
        '.theme-doc-markdown',
        'article',
        'main article',
        'main .container',
        '.markdown',
        'main'
      ];

      let contentArea = null;
      for (const selector of selectors) {
        contentArea = document.querySelector(selector);
        if (contentArea) break;
      }

      if (contentArea && !document.getElementById('translation-button-container')) {
        // Add button at the beginning of the content area
        if (contentArea.firstChild) {
          contentArea.insertBefore(buttonContainer, contentArea.firstChild);
        } else {
          contentArea.appendChild(buttonContainer);
        }
      } else {
        // Fallback: Add to body if no content area found
        if (!document.getElementById('translation-button-container')) {
          document.body.appendChild(buttonContainer);
        }
      }
    };

    const translatePageContent = async () => {
      // In a real implementation, this would call a translation API
      // For now, we'll simulate by storing original content and showing placeholder

      // Store original text content
      const contentElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, td, th, span, div');
      const newOriginalContent: Record<string, string> = {};

      contentElements.forEach((element, index) => {
        const elementId = `original_${index}`;
        newOriginalContent[elementId] = element.textContent || '';
        // In a real implementation, we would call the translation API here
        // For now, we'll just show that translation would happen
        element.setAttribute('data-original-text', element.textContent || '');
        // Simulate translated text with placeholder
        element.setAttribute('data-translated-text', `[Urdu translation of: ${(element.textContent || '').substring(0, 20)}...]`);
      });

      setOriginalContent(newOriginalContent);

      // Show a notification that translation would happen via API
      alert('In a full implementation, this would translate the page content to Urdu using a translation API.');
    };

    const restoreOriginalContent = () => {
      // Restore all text elements to their original state
      const contentElements = document.querySelectorAll('[data-original-text]');
      contentElements.forEach(element => {
        const originalText = element.getAttribute('data-original-text');
        if (originalText) {
          element.textContent = originalText;
        }
      });
    };

    // Add the button when the component mounts
    addTranslationButton();

    // For SPA navigation in Docusaurus, re-inject the button when route changes
    let currentPath = window.location.pathname;
    const observer = new MutationObserver(() => {
      if (window.location.pathname !== currentPath) {
        currentPath = window.location.pathname;
        // Wait a bit for new content to load, then inject button again
        setTimeout(() => {
          setIsTranslated(false);
          injectButton();
        }, 500);
      }
    });

    observer.observe(document, { childList: true, subtree: true });

    // Cleanup observer on unmount
    return () => {
      observer.disconnect();
    };
  }, [isTranslated]);

  return null; // This component doesn't render anything itself
};

export default TranslationProvider;