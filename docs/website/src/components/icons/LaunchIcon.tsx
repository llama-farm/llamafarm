import React from 'react'

export default function LaunchIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 32 32"
      fill="none"
      {...props}
    >
      <path
        d="M26 28H6C5.46979 27.9993 4.96149 27.7883 4.58658 27.4134C4.21166 27.0385 4.00071 26.5302 4 26V6C4.00071 5.46979 4.21166 4.96149 4.58658 4.58658C4.96149 4.21166 5.46979 4.00071 6 4H16V6H6V26H26V16H28V26C27.9993 26.5302 27.7883 27.0385 27.4134 27.4134C27.0385 27.7883 26.5302 27.9993 26 28V28Z"
        fill="currentColor"
      />
      <path
        d="M20 2V4H26.586L18 12.586L19.414 14L28 5.414V12H30V2H20Z"
        fill="currentColor"
      />
    </svg>
  )
}
