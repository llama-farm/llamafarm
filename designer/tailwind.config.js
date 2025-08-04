/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
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
    },
  },
  plugins: [],
} 