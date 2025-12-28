import React, { useState } from 'react';
import clsx from 'clsx';
import {useAuth} from '../../contexts/AuthContext';
import styles from './Auth.module.css';

interface SignInProps {
  onSwitchToSignUp?: () => void;
}

const SignIn: React.FC<SignInProps> = ({ onSwitchToSignUp }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      // In a real app, you would send this data to your backend
      console.log('Sign In:', { email, password, rememberMe });

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Update authentication state
      login();

      alert('Sign in successful! Redirecting to home page...');
      // Redirect to home page after successful sign in using window.location
      // Use the full base URL path that Docusaurus is deployed under
      window.location.href = '/';
    } catch (error) {
      console.error('Sign in error:', error);
      alert('Sign in failed. Please check your credentials and try again.');
    }
  };

  return (
    <div className={clsx(styles.authContainer, styles.signIn)}>
      <div className={styles.authCard}>
        <div className={styles.authHeader}>
          <h2>Sign In</h2>
          <p>Welcome back! Please sign in to continue</p>
        </div>

        <form onSubmit={handleSubmit} className={styles.authForm}>
          <div className={styles.inputGroup}>
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required
            />
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />
          </div>

          <div className={styles.formOptions}>
            <label className={styles.checkboxContainer}>
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <span className={styles.checkmark}></span>
              Remember me
            </label>

            <a href="#" className={styles.forgotPassword}>Forgot Password?</a>
          </div>

          <button type="submit" className={styles.authButton}>
            Sign In
          </button>
        </form>

        <div className={styles.authFooter}>
          <p>
            Don't have an account?{' '}
            <button
              type="button"
              onClick={onSwitchToSignUp}
              className={styles.switchLink}
            >
              Sign Up
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignIn;