import React from 'react';
import Layout from '@theme/Layout';
import AuthPage from '../components/Auth';

export default function SignInPage(): JSX.Element {
  return (
    <Layout title="Sign In" description="Sign in to your account">
      <AuthPage />
    </Layout>
  );
}