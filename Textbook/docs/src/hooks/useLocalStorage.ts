import { useState, useEffect } from 'react';

function getSSRSafeValue(key: string, defaultValue: string): string | null {
  if (typeof window === 'undefined') {
    return defaultValue;
  }
  return localStorage.getItem(key);
}

function setSSRSafeValue(key: string, value: string): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem(key, value);
  }
}

export function useLocalStorage(key: string, defaultValue: string): [string | null, (value: string) => void] {
  const [storedValue, setStoredValue] = useState<string | null>(() => getSSRSafeValue(key, defaultValue));

  const setValue = (value: string) => {
    try {
      setStoredValue(value);
      setSSRSafeValue(key, value);
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  };

  useEffect(() => {
    // Update state when the key changes in localStorage (from other tabs)
    if (typeof window !== 'undefined') {
      const handleStorageChange = () => {
        setStoredValue(getSSRSafeValue(key, defaultValue));
      };

      window.addEventListener('storage', handleStorageChange);
      return () => window.removeEventListener('storage', handleStorageChange);
    }
  }, [key, defaultValue]);

  return [storedValue, setValue];
}