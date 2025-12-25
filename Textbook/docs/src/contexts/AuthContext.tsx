import React, { createContext, useContext, ReactNode, useEffect } from 'react';
import { storage } from '../utils/storageUtils';

interface AuthContextType {
  isAuthenticated: boolean;
  login: () => void;
  logout: () => void;
  signup: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = React.useState<boolean>(false); // Default to false during SSR
  const [isClient, setIsClient] = React.useState<boolean>(false);

  useEffect(() => {
    // Set isClient to true on mount (client-side only)
    setIsClient(true);

    // Load authentication status from localStorage after component mounts (client-side only)
    const storedAuthStatus = storage.getItem('isAuthenticated');
    if (storedAuthStatus) {
      setIsAuthenticated(storedAuthStatus === 'true');
    }
  }, []);

  useEffect(() => {
    // Update localStorage when authentication status changes (client-side only)
    if (isClient) {
      storage.setItem('isAuthenticated', isAuthenticated.toString());
    }
  }, [isAuthenticated, isClient]);

  const login = () => {
    setIsAuthenticated(true);
  };

  const signup = () => {
    setIsAuthenticated(true);
  };

  const logout = () => {
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, logout, signup }}>
      {children}
    </AuthContext.Provider>
  );
};