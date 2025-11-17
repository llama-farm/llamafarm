import React from 'react';

// Custom Root component to inject GTM noscript fallback into body
export default function Root({children}) {
  return (
    <>
      {/* Google Tag Manager (noscript) - must be immediately after opening <body> tag */}
      <noscript>
        <iframe
          src="https://www.googletagmanager.com/ns.html?id=GTM-KQZL3NKC"
          height="0"
          width="0"
          style={{display: 'none', visibility: 'hidden'}}
        />
      </noscript>
      {children}
    </>
  );
}
