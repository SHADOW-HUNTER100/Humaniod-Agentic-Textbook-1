import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import type {Props} from '@theme/Layout';
import { AuthProvider } from '../contexts/AuthContext';
import TranslationProvider from '../components/TranslateButton/TranslationProvider';

export default function Layout(props: Props) {
  return (
    <AuthProvider>
      <OriginalLayout {...props} />
      <TranslationProvider />
    </AuthProvider>
  );
}