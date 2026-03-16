/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Status colors - muted for night ops, lower contrast
        status: {
          good: '#16703a',
          warning: '#8a5c08',
          critical: '#8a2424',
        },
        // Detection pin colors - very muted for night ops
        detection: {
          person: '#6b3333',
          vehicle: '#6b5533',
          animal: '#336b4d',
        },
        // Dark navy palette - command center at 2am
        surface: {
          base: '#0d1b2a',
          raised: '#1b2838',
          overlay: '#243447',
          border: '#1e2d3d',
          bar: '#0f1922',
        },
        // Accent
        accent: {
          DEFAULT: '#3d7cc7',
          muted: '#2a5a8f',
        },
        // Text
        text: {
          primary: '#b8c4d0',
          secondary: '#6b7a8a',
          dim: '#4a5868',
        },
        // Kill button
        kill: {
          DEFAULT: '#7a1a1a',
          hover: '#8a2424',
        },
        // Arm state
        arm: {
          armed: '#c4820a',
          disarmed: '#4a5868',
        },
      },
      fontSize: {
        'readable': ['14px', '1.4'],
      },
      spacing: {
        'touch': '44px',
      },
      animation: {
        'pulse-detection': 'pulse-detection 1.5s ease-in-out infinite',
        'slide-in': 'slide-in 0.3s ease-out',
        'fade-in': 'fade-in 0.2s ease-out',
        'glow-armed': 'glow-armed 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-detection': {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.3)', opacity: '0.7' },
        },
        'slide-in': {
          '0%': { transform: 'translateY(-100%)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'glow-armed': {
          '0%, 100%': { boxShadow: '0 0 4px rgba(196, 130, 10, 0.3)' },
          '50%': { boxShadow: '0 0 12px rgba(196, 130, 10, 0.6)' },
        }
      }
    },
  },
  plugins: [],
}
