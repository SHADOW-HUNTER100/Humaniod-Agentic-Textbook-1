// Safe storage utility that handles SSR properly
export const storage = {
  getItem: (key: string): string | null => {
    if (typeof window === 'undefined') {
      return null;
    }
    try {
      return window.localStorage.getItem(key);
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
      return null;
    }
  },

  setItem: (key: string, value: string): void => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.setItem(key, value);
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  },

  removeItem: (key: string): void => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.removeItem(key);
    } catch (error) {
      console.warn(`Error removing localStorage key "${key}":`, error);
    }
  },

  clear: (): void => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.clear();
    } catch (error) {
      console.warn('Error clearing localStorage:', error);
    }
  },
};