import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import Link from '@docusaurus/Link';

const NavbarAuthButtons: React.FC = () => {
  const { isAuthenticated, logout } = useAuth();

  const handleLogout = () => {
    logout();
    window.location.href = '/';
  };

  if (isAuthenticated) {
    return (
      <Link
        className="navbar__item navbar__link"
        to="#"
        onClick={(e) => {
          e.preventDefault();
          handleLogout();
        }}
      >
        Logout
      </Link>
    );
  } else {
    return (
      <>
        <Link
          className="navbar__item navbar__link"
          to="/sign-in"
        >
          Sign In
        </Link>
        <Link
          className="navbar__item navbar__link"
          to="/sign-up"
        >
          Sign Up
        </Link>
      </>
    );
  }
};

export default React.memo(NavbarAuthButtons);