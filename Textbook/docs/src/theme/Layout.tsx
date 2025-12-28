import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import type {Props} from '@theme/Layout';
import { AuthProvider } from '../contexts/AuthContext';

export default function Layout(props: Props) {
  return (
    <AuthProvider>
      <OriginalLayout {...props} />
    </AuthProvider>
  );
}