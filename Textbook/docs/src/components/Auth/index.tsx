import React, { useState } from 'react';
import clsx from 'clsx';
import SignIn from './SignIn';
import SignUp from './SignUp';
import styles from './Auth.module.css';

const AuthPage: React.FC = () => {
  const [isSignIn, setIsSignIn] = useState(true);

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