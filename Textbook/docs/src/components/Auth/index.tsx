import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import { useLocation } from '@docusaurus/router';
import SignIn from './SignIn';
import SignUp from './SignUp';
import styles from './Auth.module.css';

const AuthPage: React.FC = () => {
  const location = useLocation();
  const [isSignIn, setIsSignIn] = useState(location.pathname.includes('sign-in'));

  // Update the state when the location changes
  useEffect(() => {
    setIsSignIn(location.pathname.includes('sign-in'));
  }, [location.pathname]);

  return (
    <div className={styles.authContainer}>
      {isSignIn ? (
        <SignIn onSwitchToSignUp={() => setIsSignIn(false)} />
      ) : (
        <SignUp onSwitchToSignIn={() => setIsSignIn(true)} />
      )}
    </div>
  );
};

export default AuthPage;