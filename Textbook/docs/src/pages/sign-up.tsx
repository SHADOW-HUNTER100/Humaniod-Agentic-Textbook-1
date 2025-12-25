import React from 'react';
import Layout from '@theme/Layout';
import AuthPage from '../components/Auth';

export default function SignUpPage(): JSX.Element {
  return (
    <Layout title="Sign Up" description="Create a new account">
      <AuthPage />
    </Layout>
  );
}