import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import type {Props} from '@theme/Layout';

export default function Layout(props: Props) {
  return (
    <OriginalLayout {...props} />
  );
}