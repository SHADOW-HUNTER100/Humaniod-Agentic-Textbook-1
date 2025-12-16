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

        <div className={styles.divider}>
          <span>OR</span>
        </div>

        <div className={styles.socialAuth}>
          <button
            className={clsx(styles.socialButton, styles.google)}
            onClick={async (e) => {
              e.preventDefault();
              try {
                // Simulate Google OAuth flow
                console.log('Initiating Google sign in');
                await new Promise(resolve => setTimeout(resolve, 1000));
                alert('Google sign in initiated! In a real app, this would redirect to Google OAuth.');
              } catch (error) {
                console.error('Google sign in error:', error);
                alert('Google sign in failed. Please try again.');
              }
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M22.56 12.25C22.56 11.47 22.49 10.72 22.36 10H12V14.26H17.92C17.66 15.63 16.88 16.79 15.71 17.57V20.34H19.28C21.36 18.42 22.56 15.6 22.56 12.25Z" fill="#4285F4"/>
              <path d="M12 23C14.97 23 17.46 22.02 19.28 20.34L15.71 17.57C14.73 18.23 13.48 18.64 12 18.64C9.14 18.64 6.71 16.69 5.84 14.09H2.18V16.96C4 20.53 7.7 23 12 23Z" fill="#34A853"/>
              <path d="M5.84 14.09C5.62 13.44 5.49 12.74 5.49 12S5.62 10.56 5.84 9.91V7.04H2.18C1.43 8.55 1 10.22 1 12S1.43 15.45 2.18 16.96L5.84 14.09Z" fill="#FBBC05"/>
              <path d="M12 5.36C13.62 5.36 15.06 5.93 16.21 7.04L19.36 4.04C17.45 2.24 14.97 1 12 1C7.7 1 4 3.47 2.18 7.04L5.84 9.91C6.71 7.31 9.14 5.36 12 5.36Z" fill="#EA4335"/>
            </svg>
            Sign in with Google
          </button>
          <button
            className={clsx(styles.socialButton, styles.github)}
            onClick={async (e) => {
              e.preventDefault();
              try {
                // Simulate GitHub OAuth flow
                console.log('Initiating GitHub sign in');
                await new Promise(resolve => setTimeout(resolve, 1000));
                alert('GitHub sign in initiated! In a real app, this would redirect to GitHub OAuth.');
              } catch (error) {
                console.error('GitHub sign in error:', error);
                alert('GitHub sign in failed. Please try again.');
              }
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M12 0C5.37 0 0 5.37 0 12C0 17.31 3.435 21.795 8.205 23.385C8.805 23.49 9.03 23.13 9.03 22.815V20.52C5.73 21.195 5.07 19.095 5.07 19.095C4.41 17.415 3.615 17.01 3.615 17.01C2.31 16.125 3.705 16.14 3.705 16.14C5.07 16.23 5.82 17.46 5.82 17.46C6.945 19.395 8.865 18.855 9.51 18.555C9.615 17.715 9.945 17.07 10.305 16.665C7.395 16.335 4.32 15.27 4.32 10.845C4.32 9.6 4.755 8.55 5.475 7.695C5.355 7.405 4.95 6.195 5.595 4.545C5.595 4.545 6.63 4.215 8.985 5.82C9.975 5.535 11.01 5.395 12.045 5.395C13.08 5.395 14.115 5.535 15.105 5.82C17.46 4.215 18.495 4.545 18.495 4.545C19.14 6.195 18.735 7.405 18.615 7.695C19.335 8.55 19.77 9.6 19.77 10.845C19.77 15.285 16.695 16.335 13.785 16.665C14.265 17.085 14.7 17.775 14.7 18.75V22.815C14.7 23.13 14.925 23.505 15.54 23.385C20.355 21.795 23.745 17.31 23.745 12C23.745 5.37 18.375 0 12 0Z" fill="currentColor"/>
            </svg>
            Sign in with GitHub
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignIn;