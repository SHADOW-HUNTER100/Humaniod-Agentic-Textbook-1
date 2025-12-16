import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import {AuthProvider} from '../contexts/AuthContext';
import type {Props} from '@theme/Layout';

export default function Layout(props: Props) {
  return (
    <AuthProvider>
      <OriginalLayout {...props} />
    </AuthProvider>
  );
}