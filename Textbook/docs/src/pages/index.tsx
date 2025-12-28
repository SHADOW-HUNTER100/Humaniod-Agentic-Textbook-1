import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import { useAuth } from '../contexts/AuthContext';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const { isAuthenticated, logout } = useAuth();

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          {isAuthenticated ? (
            <>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                LETS LEARN QUICKILY
              </Link>
              <Link
                className="button button--primary button--lg margin-left--sm"
                to="/"
                onClick={(e) => {
                  e.preventDefault();
                  logout();
                  window.location.href = '/';
                }}>
                Logout
              </Link>
            </>
          ) : (
            <>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                LETS LEARN QUICKILY
              </Link>
              <Link
                className="button button--primary button--lg margin-left--sm"
                to="/sign-up">
                Sign Up
              </Link>
              <Link
                className="button button--outline button--lg margin-left--sm"
                to="/sign-in">
                Sign In
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
