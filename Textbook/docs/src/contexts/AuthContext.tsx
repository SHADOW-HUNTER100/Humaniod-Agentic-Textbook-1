import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

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
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false); // Default to false during SSR
  const [isClient, setIsClient] = useState<boolean>(false);

  useEffect(() => {
    // Set isClient to true on mount (client-side only)
    setIsClient(true);

    // Load authentication status from localStorage after component mounts
    const storedAuthStatus = localStorage.getItem('isAuthenticated');
    setIsAuthenticated(storedAuthStatus === 'true');
  }, []);

  useEffect(() => {
    // Only update localStorage in the browser
    if (isClient) {
      localStorage.setItem('isAuthenticated', isAuthenticated.toString());
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