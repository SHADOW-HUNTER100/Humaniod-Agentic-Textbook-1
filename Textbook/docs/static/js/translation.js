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
  // Try multiple selectors to find the main content area
  let article = document.querySelector('article.markdown');
  if (!article) {
    article = document.querySelector('main article');
  }
  if (!article) {
    article = document.querySelector('main');
  }
  if (!article) {
    article = document.querySelector('.container');
  }

  if (article) {
    // Check if button already exists to avoid duplicates
    if (!document.getElementById('translateToUrdu')) {
      const translateButton = document.createElement('button');
      translateButton.id = 'translateToUrdu';
      translateButton.textContent = 'Translate to Urdu';
      translateButton.style.cssText = 'background: #25c2a0; color: white; border: none; padding: 10px 15px; margin-bottom: 20px; cursor: pointer; border-radius: 4px; font-weight: bold; display: block; margin-left: 0; margin-right: 0;';
      translateButton.onclick = translateToUrdu;

      // Add the button at the beginning of the content
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
}

function translateToUrdu() {
  alert('In a full implementation, this would translate the current page to Urdu');
  // This would connect to a translation API in a full implementation
}

// Initialize when the page loads
initTranslationButtons();