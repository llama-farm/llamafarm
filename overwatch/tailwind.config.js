/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Status colors - high contrast for field ops
        status: {
          good: '#22c55e',      // Green
          warning: '#f59e0b',   // Amber
          critical: '#ef4444',  // Red
        },
        // Detection pin colors
        detection: {
          person: '#ef4444',    // Red
          vehicle: '#f59e0b',   // Yellow/Amber
          animal: '#22c55e',    // Green
        },
        // Dark theme palette
        surface: {
          base: '#0a0a0a',
          raised: '#141414',
          overlay: '#1f1f1f',
          border: '#2a2a2a',
          bar: '#141414',
        },
        navy: {
          50: '#e0e6ed',
          100: '#c1cdd9',
          200: '#8899aa',
          300: '#5a7a94',
          400: '#3a5a74',
          500: '#1f1f1f',
          600: '#141414',
          700: '#141414',
          800: '#0a0a0a',
          900: '#141414',
        }
      },
      fontSize: {
        // Minimum 14px for readability
        'readable': ['14px', '1.4'],
      },
      spacing: {
        // Min 44px touch targets
        'touch': '44px',
      },
      animation: {
        'pulse-detection': 'pulse-detection 1.5s ease-in-out infinite',
        'slide-in': 'slide-in 0.3s ease-out',
      },
      keyframes: {
        'pulse-detection': {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.3)', opacity: '0.7' },
        },
        'slide-in': {
          '0%': { transform: 'translateY(-100%)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        }
      }
    },
  },
  plugins: [],
}
