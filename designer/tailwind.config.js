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
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        chart: {
          1: 'hsl(var(--chart-1))',
          2: 'hsl(var(--chart-2))',
          3: 'hsl(var(--chart-3))',
          4: 'hsl(var(--chart-4))',
          5: 'hsl(var(--chart-5))',
        },
      },
      animation: {
        'scroll-up': 'scroll-up 10s linear infinite',
      },
      keyframes: {
        'scroll-up': {
          '0%': {
            transform: 'translateY(0%)',
          },
          '100%': {
            transform: 'translateY(-50%)',
          },
        },
      },
      fontFamily: {
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
        serif: [
          '"IBM Plex Sans"',
          'ui-serif',
          'Georgia',
          'Cambria',
          'Times New Roman',
          'serif',
        ],
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
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
    },
  },
  plugins: [
    require('tailwindcss-animate'),
    // Custom scrollbar utilities plugin
    function({ addUtilities, theme }) {
      const scrollbarUtilities = {
        // Thin scrollbar for general use
        '.scrollbar-thin': {
          'scrollbar-width': 'thin',
          'scrollbar-color': 'hsl(var(--border)) hsl(var(--muted))',
          '&::-webkit-scrollbar': {
            width: '6px',
            height: '6px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'transparent',
          },
          '&::-webkit-scrollbar-thumb': {
            'background-color': 'rgba(100, 100, 100, 0.5)',
            'border-radius': '6px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            'background-color': 'rgba(100, 100, 100, 0.7)',
          },
        },
        // Custom scrollbar for editors
        '.scrollbar-custom': {
          'scrollbar-width': 'thin',
          'scrollbar-color': 'hsl(var(--border)) hsl(var(--muted))',
          '&::-webkit-scrollbar': {
            width: '12px',
          },
          '&::-webkit-scrollbar-track': {
            'background-color': 'hsl(var(--muted))',
          },
          '&::-webkit-scrollbar-thumb': {
            'background-color': 'hsl(var(--border))',
            'border-radius': '6px',
            border: '2px solid hsl(var(--muted))',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            'background-color': 'hsl(var(--muted-foreground) / 0.3)',
          },
        },
        // Dark mode specific scrollbar
        '.dark .scrollbar-custom': {
          'scrollbar-color': '#3e4451 #1c2028',
          '&::-webkit-scrollbar-track': {
            background: '#1c2028',
          },
          '&::-webkit-scrollbar-thumb': {
            background: '#3e4451',
            border: '2px solid #1c2028',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            background: '#5c6370',
          },
        },
      }
      
      addUtilities(scrollbarUtilities)
    }
  ],
}
