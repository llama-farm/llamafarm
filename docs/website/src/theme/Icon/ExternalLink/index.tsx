import React from 'react';
import LaunchIcon from '../../../components/icons/LaunchIcon';
import type {Props} from '@theme/Icon/ExternalLink';

export default function IconExternalLink(props: Props): JSX.Element {
  const defaultStyle = {
    width: '1rem',
    height: '1rem',
    marginLeft: '0.25rem',
    verticalAlign: 'middle',
    opacity: 0.8,
  };
  
  return (
    <LaunchIcon
      {...props}
      style={{...defaultStyle, ...props.style}}
    />
  );
}

