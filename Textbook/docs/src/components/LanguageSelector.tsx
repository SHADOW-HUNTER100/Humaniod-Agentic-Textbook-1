import React, { useState, useEffect } from 'react';

const LanguageSelector = () => {
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [showDropdown, setShowDropdown] = useState(false);

  // Language options with codes and names
  const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Español' },
    { code: 'fr', name: 'Français' },
    { code: 'de', name: 'Deutsch' },
    { code: 'it', name: 'Italiano' },
    { code: 'pt', name: 'Português' },
    { code: 'ur', name: 'اردو' }
  ];

  const handleLanguageChange = (langCode: string) => {
    setSelectedLanguage(langCode);
    setShowDropdown(false);

    // In a real implementation, you might redirect to localized routes
    // or trigger content translation via API
    console.log(`Selected language: ${langCode}`);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!(event.target as Element).closest('.language-selector')) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <div className="language-selector" style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        style={{
          background: 'none',
          border: '1px solid #ccc',
          padding: '6px 10px',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '14px'
        }}
      >
        {languages.find(lang => lang.code === selectedLanguage)?.name || 'English'} ▼
      </button>

      {showDropdown && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: '0',
          backgroundColor: 'white',
          border: '1px solid #ccc',
          borderRadius: '4px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          zIndex: 1000,
          minWidth: '120px'
        }}>
          {languages.map((lang) => (
            <div
              key={lang.code}
              onClick={() => handleLanguageChange(lang.code)}
              style={{
                padding: '8px 12px',
                cursor: 'pointer',
                borderBottom: lang.code === 'ur' ? 'none' : '1px solid #eee',
                fontSize: '14px',
                textAlign: 'left',
                direction: lang.code === 'ur' ? 'rtl' : 'ltr'
              }}
              onMouseDown={(e) => e.preventDefault()} // Prevent button blur on click
            >
              {lang.name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageSelector;