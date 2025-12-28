import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';
import GoogleTranslateService from '../../utils/googleTranslateService';

interface TranslateButtonProps {
  content?: string; // Optional content to translate (if translating specific content)
  url?: string; // Optional specific URL to translate
  className?: string;
}

const TranslateButton: React.FC<TranslateButtonProps> = ({
  content,
  url,
  className
}) => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const translateContent = async () => {
    setIsTranslating(true);
    setError(null);

    try {
      // If specific content is provided, use the API service
      if (content && GoogleTranslateService['API_KEY']) {
        try {
          const result = await GoogleTranslateService.translateText(content, 'ur');
          // In a real implementation, you would update the UI with the translated content
          // For now, we'll just show an alert
          alert('Translation would be displayed here in a full implementation');
        } catch (apiError) {
          console.error('API translation failed:', apiError);
          // Fallback to Google Translate page
          const encodedContent = encodeURIComponent(content);
          window.open(
            `https://translate.google.com/?sl=auto&tl=ur&text=${encodedContent}&op=translate`,
            '_blank'
          );
        }
      } else {
        // If no specific content or no API key, translate the current page
        const currentUrl = url || window.location.href;
        const translateUrl = GoogleTranslateService.getTranslateUrl(currentUrl, 'ur');
        window.open(translateUrl, '_blank');
      }
    } catch (err) {
      console.error('Translation error:', err);
      setError('Failed to translate content. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <div className={clsx(styles.translateContainer, className)}>
      <button
        onClick={translateContent}
        disabled={isTranslating}
        className={clsx(
          styles.translateButton,
          'button button--primary',
          {[styles.loading]: isTranslating}
        )}
        title="Translate to Urdu"
      >
        {isTranslating ? (
          <>
            <span className={styles.spinner}></span>
            Translating...
          </>
        ) : (
          <>
            <span className={styles.icon}>🇺🇷</span>
            Translate to Urdu
          </>
        )}
      </button>
      {error && (
        <div className={styles.errorMessage}>
          {error}
        </div>
      )}
    </div>
  );
};

export default TranslateButton;