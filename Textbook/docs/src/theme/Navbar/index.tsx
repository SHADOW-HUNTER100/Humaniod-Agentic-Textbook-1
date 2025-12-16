import React, {useState, useEffect} from 'react';
import OriginalNavbar from '@theme-original/Navbar';
import {useLocation} from '@docusaurus/router';
import clsx from 'clsx';
import styles from './styles.module.css';

export default function NavbarWrapper(props) {
  const location = useLocation();
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check authentication status from localStorage on mount
  useEffect(() => {
    const authStatus = localStorage.getItem('isAuthenticated') === 'true';
    setIsAuthenticated(authStatus);
  }, []);

  // Only show auth buttons on non-auth pages
  const isAuthPage = location.pathname.includes('/sign-');

  // Prepare navbar items based on authentication status
  let navbarItems = props.items ? [...props.items] : [];

  if (!isAuthPage && !isAuthenticated) {
    // Add sign in and sign up buttons for unauthenticated users
    navbarItems = [
      ...(props.items || []),
      {
        to: '/sign-in',
        label: 'Sign In',
        position: 'right',
      },
      {
        to: '/sign-up',
        label: 'Sign Up',
        position: 'right',
      }
    ];
  } else if (isAuthenticated && !isAuthPage) {
    // Add logout button for authenticated users
    navbarItems = [
      ...(props.items || []),
      {
        type: 'button',
        label: 'Logout',
        position: 'right',
        onClick: () => {
          localStorage.setItem('isAuthenticated', 'false');
          setIsAuthenticated(false);
          window.location.href = '/';
        }
      }
    ];
  }

  // Create new props with updated items
  const newProps = {
    ...props,
    items: navbarItems
  };

  return (
    <OriginalNavbar {...newProps} />
  );
}