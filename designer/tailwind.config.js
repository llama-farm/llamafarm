/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'blue-50': '#D3DDFF',
        'blue-100': '#85B1FF',
        'blue-200': '#017BF7',
        'blue-300': '#007FFF',
        'blue-400': '#0F246D',
        'blue-500': '#131E45',
        'blue-600': '#263052',
        'blue-700': '#040C1D',
        'blue-800': '#000B1B',

        'green-100': '#48FFE4',

        'gray-100': '#C6C6C6',
        'gray-200': '#F4F4F4',
        'gray-700': '#616161',

        // 'gray-50': '#F4F4F4',
      },
      animation: {
        'scroll-up': 'scroll-up 10s linear infinite',
      },
      keyframes: {
        'scroll-up': {
          '0%': { transform: 'translateY(0%)' },
          '100%': { transform: 'translateY(-50%)' },
        },
      },
      fontFamily: {
        // Use IBM Plex Sans everywhere by default
        sans: [
          '"IBM Plex Sans"',
          'ui-sans-serif',
          'system-ui',
          'Segoe UI',
          'Roboto',
          'Helvetica',
          'Arial',
          'sans-serif',
        ],
        // Map serif utility to Sans as well to keep existing components consistent
        serif: [
          '"IBM Plex Sans"',
          'ui-serif',
          'Georgia',
          'Cambria',
          'Times New Roman',
          'serif',
        ],
        // Use IBM Plex Mono for code-like inputs/blocks
        mono: [
          '"IBM Plex Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'monospace',
        ],
      },
    },
  },
  plugins: [],
}
