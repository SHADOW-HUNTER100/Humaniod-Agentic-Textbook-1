import React, { createContext, useContext, ReactNode, useEffect } from 'react';
import { useLocalStorage } from '../hooks/useLocalStorage';

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
  const [storedAuthStatus, setStoredAuthStatus] = useLocalStorage('isAuthenticated', 'false');
  const [isAuthenticated, setIsAuthenticated] = React.useState<boolean>(() => storedAuthStatus === 'true');
  const [isClient, setIsClient] = React.useState<boolean>(false);

  useEffect(() => {
    // Set isClient to true on mount (client-side only)
    setIsClient(true);
  }, []);

  useEffect(() => {
    // Update state when localStorage changes
    setIsAuthenticated(storedAuthStatus === 'true');
  }, [storedAuthStatus]);

  useEffect(() => {
    // Update localStorage when authentication status changes
    if (isClient) {
      setStoredAuthStatus(isAuthenticated.toString());
    }
  }, [isAuthenticated, isClient, setStoredAuthStatus]);

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