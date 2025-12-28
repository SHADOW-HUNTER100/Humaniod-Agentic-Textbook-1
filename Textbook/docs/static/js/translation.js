// Translation functionality for chapters
function initTranslationButtons() {
  // Add translation button to each chapter
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addTranslateButton);
  } else {
    addTranslateButton();
  }
}

function addTranslateButton() {
  // Wait a bit for the page to fully load and render
  setTimeout(() => {
    // Try multiple selectors to find the main content area in Docusaurus v3
    let selectors = [
      'article', // Generic article tag
      'main article', // Main content area with article
      'main .container', // Container inside main
      'main > .container', // Direct container child of main
      '.theme-doc-markdown', // Docusaurus v3 specific class
      '.markdown', // Markdown content area
      '.docMainContainer', // Docusaurus documentation container
      '.docPage', // Docusaurus page container
      'main', // Fallback to main
      '.container', // Fallback to container
    ];

    let article = null;
    for (let selector of selectors) {
      article = document.querySelector(selector);
      if (article) break;
    }

    if (article) {
      // Check if button already exists to avoid duplicates
      if (!document.getElementById('translateToUrdu')) {
        const translateButton = document.createElement('div'); // Use div wrapper for better positioning
        translateButton.id = 'translateToUrduWrapper';
        translateButton.style.cssText = 'margin-bottom: 20px; text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;';

        const button = document.createElement('button');
        button.id = 'translateToUrdu';
        button.textContent = 'Translate to Urdu';
        button.style.cssText = 'background: linear-gradient(90deg, #4285f4, #34a853, #fbbc05, #ea4335); color: white; border: none; padding: 12px 24px; cursor: pointer; border-radius: 24px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); transition: all 0.2s ease;';

        button.onmouseover = function() {
          this.style.transform = 'translateY(-2px)';
          this.style.boxShadow = '0 6px 16px rgba(0,0,0,0.3)';
        };

        button.onmouseout = function() {
          this.style.transform = 'translateY(0)';
          this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
        };

        button.onclick = translateToUrdu;

        translateButton.appendChild(button);
        // Add the button at the START of the content (as requested)
        if (article.firstChild) {
          article.insertBefore(translateButton, article.firstChild);
        } else {
          article.appendChild(translateButton);
        }
      }
    } else {
      // If no specific article is found, add to body as fallback
      if (!document.getElementById('translateToUrdu')) {
        const translateButton = document.createElement('button');
        translateButton.id = 'translateToUrdu';
        translateButton.textContent = 'Translate to Urdu';
        translateButton.style.cssText = 'position: fixed; top: 10px; right: 10px; background: #25c2a0; color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 4px; font-weight: bold; z-index: 1000;';
        translateButton.onclick = translateToUrdu;

        document.body.appendChild(translateButton);
      }
    }
  }, 1000); // Wait 1 second for content to load
}

function translateToUrdu() {
  // Check if content is already translated
  const button = document.getElementById('translateToUrdu');
  if (!button) return;

  const isCurrentlyTranslated = button.textContent === 'Show Original';

  if (isCurrentlyTranslated) {
    // Show original content
    location.reload(); // Simple approach: reload to show original
  } else {
    // For a full implementation, we would need Google Translate API
    // Since we don't have an API key, we'll demonstrate by simulating the translation
    // In a real implementation, you would need to set up Google Cloud Translation API

    // Show a message to the user about the API requirement
    alert('To translate content to Urdu, you need to set up Google Cloud Translation API. The page will open in Google Translate.');

    // Fallback: Open in Google Translate
    const currentUrl = window.location.href;
    const translateUrl = `https://translate.google.com/translate?hl=ur&sl=auto&tl=ur&u=${encodeURIComponent(currentUrl)}`;
    window.open(translateUrl, '_blank');
  }
}

// Initialize when the page loads
initTranslationButtons();