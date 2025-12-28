import React, { useState } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import {useAuth} from '../../contexts/AuthContext';
import styles from './Auth.module.css';

interface SignUpProps {
  onSwitchToSignIn?: () => void;
}

const SignUp: React.FC<SignUpProps> = ({ onSwitchToSignIn }) => {
  const { signup } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      alert('Passwords do not match');
      return;
    }

    if (!agreedToTerms) {
      alert('You must agree to the Terms of Service and Privacy Policy');
      return;
    }

    try {
      // In a real app, you would send this data to your backend
      console.log('Sign Up:', { name, email, password, agreedToTerms });

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Update authentication state
      signup();

      alert('Sign up successful! Redirecting to home page...');
      // Redirect to home page after successful sign up using window.location
      // Use the full base URL path that Docusaurus is deployed under
      window.location.href = '/';
    } catch (error) {
      console.error('Sign up error:', error);
      alert('Sign up failed. Please try again.');
    }
  };

  return (
    <div className={clsx(styles.authContainer, styles.signUp)}>
      <div className={styles.authCard}>
        <div className={styles.authHeader}>
          <h2>Sign Up</h2>
          <p>Create an account to get started</p>
        </div>

        <form onSubmit={handleSubmit} className={styles.authForm}>
          <div className={styles.inputGroup}>
            <label htmlFor="name">Full Name</label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your full name"
              required
            />
          </div>

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
              placeholder="Create a password"
              required
            />
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="confirmPassword">Confirm Password</label>
            <input
              type="password"
              id="confirmPassword"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Confirm your password"
              required
            />
          </div>

          <label className={clsx(styles.checkboxContainer, styles.terms)}>
            <input
              type="checkbox"
              checked={agreedToTerms}
              onChange={(e) => setAgreedToTerms(e.target.checked)}
              required
            />
            <span className={styles.checkmark}></span>
            I agree to the <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>
          </label>

          <button type="submit" className={styles.authButton}>
            Sign Up
          </button>
        </form>

        <div className={styles.authFooter}>
          <p>
            Already have an account?{' '}
            <button
              type="button"
              onClick={onSwitchToSignIn}
              className={styles.switchLink}
            >
              Sign In
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignUp;